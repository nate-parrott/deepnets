from net import Net, batch_generator
import tensorflow as tf
import numpy as np
import random
from util import weight_var, load_image, leaky_relu, one_hot
from util import show_image_grayscale  as show_image
import os
from tensorflow.contrib.layers.python.layers import batch_norm
from hw0 import read_images, read_labels

def load_dataset(name, noise=False):
    def featurize(image):
        return image.flatten().astype(float) / 255.0
    images = np.array([featurize(image) for image in read_images(name + '-images.idx3-ubyte')])
    labels = np.array([one_hot(label, 11) for label in read_labels(name + '-labels.idx1-ubyte')])
    extra_count = int(labels.shape[0] * 0.1)
    images = np.append(images, np.random.rand(extra_count, 784), axis=0)
    labels = np.append(labels, np.array([one_hot(10,11) for _ in xrange(extra_count)]), axis=0)
    # print images.shape, labels.shape
    return images, labels

class GAN(Net):
    input_size = 28
    noise_size = 64
    last_accuracy = None
    
    def setup(self):        
        real_image_input = tf.placeholder(tf.float32, [None, self.input_size ** 2], name='real_image_input')
        noise_input = tf.placeholder(tf.float32, [None, self.noise_size], name='noise_input')
        self.noise_input = noise_input
        use_real_image = tf.placeholder(tf.float32, [None], name='use_real_input') # 1 if using real image, 0 if using noise
        disc_dropout_keep_prob = tf.placeholder(tf.float32, name='disc_dropout_keep_prob')        
                
        patch_size = 5
        
        def create_conv(input, in_channels, out_channels, weight_set=[], patch_size=2):
            input = batch_norm(input, variables_collections=[weight_set])
            w = weight_var([patch_size, patch_size, in_channels, out_channels])
            b = weight_var([out_channels])
            weight_set.append(w)
            weight_set.append(b)
            
            conv = tf.nn.conv2d(input, w, strides=[1,2,2,1], padding='SAME')
            activation = leaky_relu(conv + b)
            return activation
        
        def create_deconv(input, in_channels, out_channels, input_image_size, weight_set=[], no_bias=False, use_weight_bias_pair=None, patch_size=2):
            # # reverse max-pooling using densely connected layer:
            # dense = tf.reshape(input, [-1, in_channels * image_size**2])
            # new_image_size = image_size*2
            # w1 = weight_var([in_channels * image_size**2, in_channels * new_image_size**2])
            # img = tf.reshape(tf.matmul(dense, w1), [-1, new_image_size, new_image_size, in_channels])
            #            
            input = batch_norm(input, variables_collections=weight_set)
                        
            if use_weight_bias_pair:
                w, b = use_weight_bias_pair
            else:
                w = weight_var([patch_size, patch_size, out_channels, in_channels])
                weight_set.append(w)
                b = weight_var([out_channels], init_zero=no_bias)
                if not no_bias:
                    weight_set.append(b)
            
            batch_size = tf.shape(input)[0]
            output_shape = tf.pack([batch_size, input_image_size*2, input_image_size*2, out_channels])
            
            deconv = tf.nn.conv2d_transpose(input, w, output_shape, strides=[1,2,2,1], padding='SAME')
            return leaky_relu(deconv + b)
        
        def create_dense(input, in_size, out_size, weight_set=[], relu=True, no_bias=False, disc_dropout=False):
            input = batch_norm(input, variables_collections=weight_set)
            if disc_dropout:
                input = tf.nn.dropout(input, disc_dropout_keep_prob)
            w = weight_var([in_size, out_size])
            weight_set.append(w)
            b = weight_var([out_size], init_zero=True)
            if not no_bias:
                weight_set.append(b)
            x = tf.matmul(input, w)
            return leaky_relu(x + b) if relu else x + b
        
        def create_conv_with_pooling(input, in_channels, out_channels, weight_set=[], patch_size=2):
            w = weight_var([patch_size, patch_size, in_channels, out_channels])
            b = weight_var([out_channels], init_zero=True)
            conv = tf.nn.conv2d(input, w, strides=[1,1,1,1], padding='SAME')
            activation = leaky_relu(conv + b)
            weight_set.append(w)
            weight_set.append(b)
            pooled = tf.nn.max_pool(activation, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            return pooled
        
        def mix(set1, set2, mix_bool):
            x = tf.transpose(set1) * (1.0 - mix_bool) + tf.transpose(set2) * mix_bool
            return tf.transpose(x)
        
        # create generator:
        gen_weights = []
        # noise_input = tf.Print(noise_input, [noise_input], 'noise input:')
        gen = create_dense(noise_input, self.noise_size, 7 * 7 * 8, gen_weights)
        # gen = tf.Print(gen, [gen], 'gen:')
        # gen = tf.nn.dropout(gen, 0.7)
        gen = tf.reshape(gen, [-1, 7, 7, 8]) # 7 x 7 x 8
        # gen = tf.nn.dropout(gen, 0.7)
        gen = create_deconv(gen, 8, 6, 7, gen_weights, patch_size=4) # 14 x 14 x 6
        gen = create_deconv(gen, 6, 1, 14, gen_weights, patch_size=4) # 28 x 28 x 1
        
        # create discriminator input:
        real_image_input_reshaped = tf.reshape(real_image_input, [-1, self.input_size, self.input_size, 1])
        disc_input = mix(gen, real_image_input_reshaped, use_real_image)
        self.disc_input = disc_input
        
        # create discriminator:
        disc_weights = []
        disc = create_conv_with_pooling(disc_input, 1, 8, disc_weights, patch_size=3) # now 14 x 14 x 8
        disc = create_conv_with_pooling(disc, 8, 16, disc_weights, patch_size=3) # now 7 x 7 x 16
        disc = tf.reshape(disc, [-1, 7 * 7 * 16])
        output = tf.nn.softmax(create_dense(disc, 7 * 7 * 16, 2, disc_weights, disc_dropout=True))
        desired_output = tf.pack([use_real_image, 1 - use_real_image], axis=1)
        loss = tf.reduce_mean(-tf.reduce_sum(desired_output * tf.log(output), reduction_indices=[1]))
        correct = tf.equal(tf.argmax(output, 1), tf.argmax(desired_output, 1))
        disc_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        optimizer = tf.train.AdamOptimizer(0.0001)
        disc_train_step = optimizer.minimize(loss, var_list=disc_weights)
        gen_train_step = optimizer.minimize(-loss, var_list=gen_weights)
        
        self.disc_weights = disc_weights
        self.gen_weights = gen_weights
        self.real_image_input = real_image_input
        self.use_real_image = use_real_image
        self.disc_dropout_keep_prob = disc_dropout_keep_prob
        self.disc_loss = loss
        self.disc_accuracy = disc_accuracy
        self.disc_train_step = disc_train_step
        self.gen_train_step = gen_train_step
        self.gen = gen
    
    def train(self, inp, outp):
        use_real_image = np.random.randint(0, 2, len(inp))
        # print use_real_image
        target_accuracy = 0.6
        train_gen = self.last_accuracy is None or self.last_accuracy > target_accuracy
        # train_gen = self.last_loss is None or self.last_loss < target_loss
        # print 'train gen:', train_gen
        noise = np.random.uniform(size=[len(inp), self.noise_size])
        if train_gen:
            self.session.run(self.gen_train_step, feed_dict={
                self.real_image_input: inp, 
                self.use_real_image: use_real_image, 
                self.disc_dropout_keep_prob: 1, 
                self.noise_input: noise
            })
        else:
            self.session.run(self.disc_train_step, feed_dict={
                self.real_image_input: inp, 
                self.use_real_image: use_real_image, 
                self.disc_dropout_keep_prob: 0.5, 
                self.noise_input: noise
            })
    
    def evaluate(self, inp, outp):
        noise = np.random.uniform(size=[len(inp), self.noise_size])
        use_real_image = np.random.randint(0, 2, len(inp))
        loss, accuracy = self.session.run([self.disc_loss, self.disc_accuracy], feed_dict={
            self.real_image_input: inp,
            self.use_real_image: use_real_image,
            self.disc_dropout_keep_prob: 1,
            self.noise_input: noise
        })
        print("Discriminator accuracy:", accuracy)
        self.last_accuracy = accuracy
        img = self.generate_images(count=10)[0]
        show_image(img, path="/Users/nateparrott/Desktop/gan.png")
        return loss
    
    def generate_images(self, count=10):
        # print 'generating images'
        noise = np.random.uniform(size=[count, self.noise_size])
        use_real_image = np.full([count], 0, dtype=np.int)
        # use_real_image = np.random.randint(0, 2, count)
        images = self.session.run(self.gen, feed_dict={
            self.use_real_image: use_real_image,
            self.disc_dropout_keep_prob: 1,
            self.noise_input: noise
        })
        # print 'done generating images'
        return images

n = GAN(dir_path='model/mgan-gan-tuned')

def train():
    test_in, test_out = load_dataset('t10k', noise=True)
    train_in, train_out = load_dataset('train', noise=True)
    train_batcher = batch_generator(train_in, train_in, size=64, random=True)
    test_batcher = batch_generator(test_in, test_in, size=128, random=True)
    n.training_loop(train_batcher, test_batcher, evaluation_interval=20)

def generate():
    while True:
        show_image(n.generate_images(count=1)[0])
        raw_input('press enter for more')

if __name__ == '__main__':
    import sys
    if 'generate' in sys.argv:
        generate()
    else:
        train()
