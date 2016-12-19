import tensorflow as tf
import numpy as np
from net import Net, batch_generator
from embedding import UNKNOWN, embeddings
import json
from simple import *
import random
from util import weight_var
from tokenize_str import tokenize

PADDING_POS = 'PADDING'
WINDOW_SIZE = 16
EMBEDDING_SIZE = 50

def pad_and_flatten_sents(sents):
    padding = [(UNKNOWN, PADDING_POS)] * (WINDOW_SIZE / 2)
    docs = []
    for sent in sents:
        docs += padding + sent
    docs += padding
    return docs

train_pairs = pad_and_flatten_sents(json.load(open('data/pos.train.json')))
test_pairs = pad_and_flatten_sents(json.load(open('data/pos.test.json')))
words, embedding_matrix = embeddings()
word_lookup = dict(((word, i) for i, word in enumerate(words)))

all_pos = set((pair[1] for pair in train_pairs + test_pairs))
pos_lookup = dict(((pos, i) for i, pos in enumerate(all_pos)))

def create_training_vec(idx, pairs):
    pairs = pairs[idx:idx+WINDOW_SIZE]
    unk = word_lookup[UNKNOWN]
    token_vec = np.array([word_lookup.get(word, unk) for word, pos in pairs], dtype=int)
    pos_vec = np.array([pos_lookup[pos] for word, pos in pairs], dtype=int)
    return token_vec, pos_vec

def get_training_mats(pairs, n):
    token_vecs = []
    pos_vecs = []
    for _ in xrange(n):
        while True:
            i = random.randint(0, len(pairs)-WINDOW_SIZE-1)
            if pairs[i+WINDOW_SIZE/2][1] != PADDING_POS:
                token_vec, pos_vec = create_training_vec(i, pairs)
                token_vecs.append(token_vec)
                pos_vecs.append(pos_vec)
                break
    return np.array(token_vecs), np.array(pos_vecs)

class Tagger(Net):
    def setup(self):
        text = tf.placeholder(tf.int32, [None, WINDOW_SIZE], name='text')
        pos = tf.placeholder(tf.int64, [None, WINDOW_SIZE], name='pos')
        
        embedding = tf.constant(embedding_matrix, name='embedding', dtype=tf.float32)
        text_embedded = tf.nn.embedding_lookup(embedding, text)
        
        def create_dense(input, input_size, output_size, relu=True):
            weights = weight_var([input_size, output_size])
            biases = weight_var([output_size])
            r = tf.matmul(input, weights) + biases
            return tf.nn.relu(r) if relu else r
        
        def create_conv(input, in_channels, out_channels, weight_set=[]):
            # input = batch_norm(input, variables_collections=[weight_set])
            w = weight_var([5, in_channels, out_channels])
            b = weight_var([out_channels])
            weight_set.append(w)
            weight_set.append(b)
            
            conv = tf.nn.conv1d(input, w, stride=2, padding='SAME')
            activation = tf.nn.relu(conv + b)
            return activation
        
        pos_to_predict = tf.unpack(pos, axis=1)[WINDOW_SIZE/2]
        
        h0 = tf.reshape(text_embedded, [-1, EMBEDDING_SIZE * WINDOW_SIZE])
        h1 = create_dense(h0, EMBEDDING_SIZE * WINDOW_SIZE, 128)
        pos_probs = create_dense(h1, 128, len(all_pos))
        # h1 = create_conv(text_embedded, EMBEDDING_SIZE, 32) # [WINDOW_SIZE/2, 64]
        # h2 = create_conv(h1, 32, 64) # [WINDOW_SIZE/4, 64]
        # pos_probs = create_dense(tf.reshape(h2, [-1, 64 * WINDOW_SIZE/4]), 64 * WINDOW_SIZE/4, len(all_pos))
        #
        cross_entropy = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(pos_probs, pos_to_predict))
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
        pos_predictions = tf.argmax(pos_probs, 1)
        
        correct = tf.equal(pos_predictions, pos_to_predict)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        self.text = text
        self.pos = pos
        
        self.train_step = train_step
        self.cross_entropy = cross_entropy
        self.pos_probs = pos_probs
        self.pos_predictions = pos_predictions
        self.accuracy = accuracy
    
    def train(self):
        token_mat, train_mat = get_training_mats(train_pairs, 64)
        _, cross_entropy = self.session.run([self.train_step, self.cross_entropy], feed_dict={self.pos: train_mat, self.text: token_mat})
        print "Cross entropy:", cross_entropy
    
    def evaluate(self):
        for dataset in ['train', 'test']:
            token_mat, train_mat = get_training_mats((test_pairs if dataset=='test' else train_pairs), 512)
            accuracy = self.session.run(self.accuracy, feed_dict={self.text: token_mat, self.pos: train_mat})
            print "Accuracy on {0}: {1}".format(dataset, accuracy * 100)
    
    def tag(self, text):
        tokens = tokenize(text)
        padded_tokens = [UNKNOWN] * (WINDOW_SIZE/2) + tokens + [UNKNOWN] * (WINDOW_SIZE/2)
        pairs = [(tk, PADDING_POS) for tk in padded_tokens]
        
        text_mat = []
        for i in range(len(tokens)):
            vec, _ = create_training_vec(i, pairs)
            text_mat.append(vec)
        
        text_mat = np.array(text_mat)
        output = self.session.run(self.pos_predictions, feed_dict={self.text: text_mat})
        pos_lookup_by_index = dict([(i, pos) for pos, i in pos_lookup.items()])        
        predictions = [pos_lookup_by_index[i] for i in output]
        return list(zip(tokens, predictions))

def train():
    t = Tagger(dir_path='model/pos')
    for step in xrange(1000000):
        t.train()
        if step % 250 == 0:
            t.evaluate()
            # print 'Accuracy:', t.evaluate()
            t.save(step)

def interact():
    t = Tagger(dir_path='model/pos')
    while True:
        text = raw_input(" > ")
        print t.tag(text)

if __name__ == '__main__':
    # train()
    interact()
