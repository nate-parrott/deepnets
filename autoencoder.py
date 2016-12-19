from net import Net, batch_generator
import tensorflow as tf
import numpy as np
import os
from util import weight_var, load_image, show_image
import math
import random

def float_tensor(x):
    return tf.convert_to_tensor(x, dtype=tf.float32)

bottleneck_size = 128 # 1319 emoji in dataset

class ImageAutoEncoder(Net):
    def setup(self):
        input = tf.placeholder(tf.float32, [None, 64, 64, 3])
        desired_output = tf.placeholder(tf.float32, [None, 64, 64, 3])
        dropout_keep_prob = tf.placeholder_with_default(float_tensor(1), [])
        artificial_hidden_logits = tf.placeholder_with_default(tf.convert_to_tensor(np.zeros(bottleneck_size), dtype=tf.float32), [bottleneck_size])
        use_artificial_hidden_logits = tf.placeholder_with_default(float_tensor(0), [])
        
        # patch_size = 5
        #
        # def create_conv(input, in_channels, out_channels):
        #     weights = weight_var([patch_size, patch_size, in_channels, out_channels])
        #     biases = weight_var([out_channels])
        #     conv = tf.nn.conv2d(input, weights, strides=[1,1,1,1], padding='SAME')
        #     activation = tf.nn.relu(conv + biases)
        #     # return activation
        #     pooled = tf.nn.max_pool(activation, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        #     return pooled
        #
        # def create_deconv(input, in_image_size, out_size):
        #     # reverse max-pooling:
        #     weights = weight_var(in_image_size ** 2, (in_image_size * 2) ** 2)
        #
        # def create_dense(input, input_size, output_size, relu=True):
        #     weights = weight_var([input_size, output_size])
        #     biases = weight_var([output_size])
        #     r = tf.matmul(input, weights) + biases
        #     return tf.nn.relu(r) if relu else r        
        #
        # conv1 = create_conv(input, 3, 16)
        # conv2 = create_conv(input, 16, 16)
        # size_now = (self.size/4)**2 * 16
        # conv2_reshaped = tf.reshape(conv2, [None, size_now])
        # dense = create_dense(conv2, size_now, size_now)
        # dense_dropout = tf.nn.dropout(dense, dropout_keep_prob)
        
        filters = []
        patch_size = 5
        
        def create_conv(input, in_channels, out_channels):
            w = weight_var([patch_size, patch_size, in_channels, out_channels])
            b = weight_var([out_channels])
            conv = tf.nn.conv2d(input, w, strides=[1,1,1,1], padding='SAME')
            activation = tf.nn.relu(conv + b)
            filters.append(w)
            pooled = tf.nn.max_pool(activation, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            return pooled
        
        def create_deconv(input, in_channels, out_channels, image_size):
            # # reverse max-pooling using densely connected layer:
            # dense = tf.reshape(input, [-1, in_channels * image_size**2])
            # new_image_size = image_size*2
            # w1 = weight_var([in_channels * image_size**2, in_channels * new_image_size**2])
            # img = tf.reshape(tf.matmul(dense, w1), [-1, new_image_size, new_image_size, in_channels])
            #
            img = tf.image.resize_images(input, image_size*2, image_size*2)
            
            w = filters.pop()
            b = weight_var([out_channels])
            
            batch_size = tf.shape(img)[0]
            deconv_shape = tf.pack([batch_size, image_size*2, image_size*2, out_channels])
            
            deconv = tf.nn.conv2d_transpose(img, w, deconv_shape, strides=[1,1,1,1], padding='SAME')
            return tf.nn.relu(deconv + b)
        
        def create_dense(input, in_size, out_size):
            input_dropped = tf.nn.dropout(input, dropout_keep_prob)
            w = weight_var([in_size, out_size])
            b = weight_var([out_size])
            return tf.nn.relu(tf.matmul(input_dropped, w) + b)
        
        img = create_conv(input, 3, 8) # now img is 32 x 32, 8 channels
        img = create_conv(img, 8, 8) # now img is 16 x 16, 8 channels
        # img = tf.Print(img, [img], message='IMG:')
        dense_size = 16**2 * 8
        hidden = tf.reshape(img, [-1, dense_size])
        hidden = create_dense(hidden, dense_size, bottleneck_size)
        
        mean, var = tf.nn.moments(hidden, axes=[1])
        mean = tf.reduce_mean(mean)
        var = tf.reduce_mean(var)
        
        hidden = (1 - use_artificial_hidden_logits) * hidden + use_artificial_hidden_logits * artificial_hidden_logits
        
        hidden = create_dense(hidden, bottleneck_size, dense_size)
        # hidden = tf.Print(hidden, [hidden], message='Hidden:')
        img = tf.reshape(img, [-1, 16, 16, 8])
        img = create_deconv(img, 8, 8, 16) # now img is 32 x 32, 8 channels
        img = create_deconv(img, 8, 3, 32) # now img is 64 x 64, 3 channels
        # img = tf.Print(img, [img], message='IMG2:')
        
        output = img
        output_dense = tf.reshape(img, [-1, 64**2 * 3])
        desired_output_dense = tf.reshape(desired_output, [-1, 64**2 * 3])
        # output_dense = tf.Print(output_dense, [output_dense], message='hey')
        # desired_output_dense = tf.Print(output_dense, [desired_output_dense], message='hey')
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(desired_output_dense * tf.log(output_dense), reduction_indices=[1]))
        loss = tf.reduce_mean(tf.reduce_sum(tf.pow(desired_output_dense - output_dense, 2), reduction_indices=[1]))
        learn_rate = 0.001
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)
        
        self.loss = loss
        # print loss.get_shape()
        self.train_step = train_step
        self.desired_output = desired_output
        self.output = output
        self.input = input
        self.dropout_keep_prob = dropout_keep_prob
        self.artificial_hidden_logits = artificial_hidden_logits
        self.use_artificial_hidden_logits = use_artificial_hidden_logits
        self.mean = mean
        self.var = var
    
    def train(self, in_data, out_data):
        self.session.run(self.train_step, feed_dict={self.input: in_data, self.desired_output: out_data, self.dropout_keep_prob: 0.5})
        # print 'trained'
    
    def evaluate(self, inputs, outputs):
        loss, mean, var = self.session.run([self.loss, self.mean, self.var], feed_dict={self.input: inputs, self.desired_output: outputs, self.dropout_keep_prob: 1})
        print loss, mean, var
        return loss
    
    def reconstruct(self, emoji_name):
        img = np.array([load_image(os.path.join('emoji', emoji_name), 64)])
        out = self.session.run(self.output, feed_dict={self.input: img})
        return out[0]
    
    def generate_random(self):
        mean = 0.838776
        stddev = math.sqrt(1.4425)
        # a = np.random.normal(mean, stddev, bottleneck_size)
        a = np.zeros(bottleneck_size)
        a.fill(mean)
        a[random.randint(0, bottleneck_size-1)] = mean*2
        blank_img = np.zeros((1,64,64,3))
        img = self.session.run(self.output, feed_dict={self.input: blank_img, self.artificial_hidden_logits: a, self.use_artificial_hidden_logits: 1})[0]
        show_image(img)

n = ImageAutoEncoder(dir_path='model/autoencoder1')

def train():
    image_paths = [os.path.join('emoji', name) for name in os.listdir('emoji') if name.endswith('.png')]
    images = np.array([load_image(path, 64) for path in image_paths])
    train_batcher = batch_generator(images, images, size=20, random=True)
    test_batcher = batch_generator(images, images, size=40, random=True)
    n.training_loop(train_batcher, test_batcher)

def reconstruct():
    while True:
        x = raw_input('emoji name > ')
        if len(x) == 0: break
        show_image(n.reconstruct(x + '.png'))

if __name__ == '__main__':
    # train()
    # show_image(load_image(os.path.join('emoji', 'airplane.png'), 64))    
    reconstruct()
    # for _ in xrange(5):
    #    n.generate_random()
