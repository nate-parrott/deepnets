from net import Net, batch_generator
import tensorflow as tf
import numpy as np
import random
from util import weight_var, load_image, show_image, leaky_relu
import os
from tensorflow.contrib.layers.python.layers import batch_norm

class GAN(Net):
    input_size = 64
    noise_size = 100
    last_loss = None
    
    def setup(self):        
        real_image_input = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3])
        noise_input = tf.placeholder(tf.float32, [None, self.noise_size])
        use_real_image = tf.placeholder(tf.float32, [None]) # 1 if using real image, 0 if using noise
        disc_dropout_keep_prob = tf.placeholder(tf.float32)        
        
        patch_size = 5
        
        def create_conv(input, in_channels, out_channels, weight_set=[]):
            input = batch_norm(input)
            w = weight_var([patch_size, patch_size, in_channels, out_channels])
            b = weight_var([out_channels])
            weight_set.append(w)
            weight_set.append(b)
            
            conv = tf.nn.conv2d(input, w, strides=[1,2,2,1], padding='SAME')
            activation = leaky_relu(conv + b)
            return activation
        
        def create_deconv(input, in_channels, out_channels, input_image_size, weight_set=[], no_bias=False, use_weight_bias_pair=None):
            # # reverse max-pooling using densely connected layer:
            # dense = tf.reshape(input, [-1, in_channels * image_size**2])
            # new_image_size = image_size*2
            # w1 = weight_var([in_channels * image_size**2, in_channels * new_image_size**2])
            # img = tf.reshape(tf.matmul(dense, w1), [-1, new_image_size, new_image_size, in_channels])
            #            
            input = batch_norm(input)
            
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
        
        def create_dense(input, in_size, out_size, weight_set=[], relu=True, no_bias=False):
            input = batch_norm(input)
            input_dropped = tf.nn.dropout(input, disc_dropout_keep_prob)
            w = weight_var([in_size, out_size])
            weight_set.append(w)
            b = weight_var([out_size], init_zero=no_bias)
            if not no_bias:
                weight_set.append(b)
            x = tf.matmul(input_dropped, w)
            return leaky_relu(x + b) if relu else x + b
        
        def create_conv_with_pooling(input, in_channels, out_channels, weight_set=[]):
            w = weight_var([patch_size, patch_size, in_channels, out_channels])
            b = weight_var([out_channels])
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
        patch_size = 5
        gen = create_dense(noise_input, self.noise_size, 4 * 4 * 16, gen_weights)
        gen = tf.nn.dropout(gen, 0.7)
        gen = tf.reshape(gen, [-1, 4, 4, 16]) # 4 x 4 x 16
        # gen = tf.nn.dropout(gen, 0.7)
        gen = create_deconv(gen, 16, 16, 4, gen_weights) # 8 x 8 x 16
        gen = create_deconv(gen, 16, 8, 8, gen_weights) # 16 x 16 x 8
        gen = create_deconv(gen, 8, 8, 16, gen_weights) # 32 x 32 x 8
        w, b = np.load('last_deconv.npy')
        gen = create_deconv(gen, 8, 3, 32, gen_weights, use_weight_bias_pair=(w,b)) # 64 x 64 x 3
        
        # create discriminator input:
        disc_input = mix(gen, real_image_input, use_real_image)
        self.disc_input = disc_input
        
        # create discriminator:
        disc_weights = []
        disc = create_conv_with_pooling(disc_input, 3, 8, disc_weights) # now 32 x 32 x 8
        disc = create_conv_with_pooling(disc, 8, 8, disc_weights) # now 16 x 16 x 8
        disc = tf.nn.dropout(disc, disc_dropout_keep_prob)
        disc = create_conv(disc, 8, 16, disc_weights) # now 8 x 8 x 16
        disc = tf.nn.dropout(disc, disc_dropout_keep_prob)
        disc = create_conv(disc, 16, 8, disc_weights) # now 4 x 4 x 8
        disc = tf.reshape(disc, [-1, 4*4*8])
        output = tf.sigmoid(create_dense(disc, 4*4*8, 1, disc_weights))
        # output = tf.Print(output, [output, use_real_image])
        # loss = tf.reduce_mean(tf.pow(output - use_real_image, 2))
        loss = tf.reduce_mean(tf.exp(output - use_real_image))
        # loss = tf.Print(loss, [output - use_real_image, use_real_image])
        
        # disc_loss = tf.reduce_mean(output - use_real_image) # tf.reduce_sum(tf.exp(output - use_real_image))
        optimizer = tf.train.AdamOptimizer(0.0001)
        disc_train_step = optimizer.minimize(loss, var_list=disc_weights)
        gen_train_step = optimizer.minimize(-loss, var_list=gen_weights)
        
        self.disc_weights = disc_weights
        self.gen_weights = gen_weights
        self.real_image_input = real_image_input
        self.noise_input = noise_input
        self.use_real_image = use_real_image
        self.disc_dropout_keep_prob = disc_dropout_keep_prob
        self.disc_loss = loss
        self.disc_train_step = disc_train_step
        self.gen_train_step = gen_train_step
        self.gen = gen
    
    def train(self, inp, outp):
        use_real_image = np.random.randint(0, 2, len(inp))
        # print use_real_image
        # target_loss = 0.1 # 0.5
        train_gen = False # random.random() < 0.75 # self.last_loss is None or self.last_loss < target_loss
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
        loss = self.session.run(self.disc_loss, feed_dict={
            self.real_image_input: inp,
            self.use_real_image: use_real_image,
            self.disc_dropout_keep_prob: 1,
            self.noise_input: noise
        })
        self.last_loss = loss
        img = self.generate_images(count=1)[0]
        show_image(img, path="/Users/nateparrott/Desktop/gan.png")
        return loss
    
    def generate_images(self, count=10):
        noise = np.random.uniform(size=[count, self.noise_size])
        use_real_image = np.full([count], 0)
        images = self.session.run(self.gen, feed_dict={
            # self.real_image_input: noise,
            self.use_real_image: use_real_image,
            self.disc_dropout_keep_prob: 1,
            self.noise_input: noise
        })
        return images
    
    def generate_images2(self, count=10):
        image_paths = [os.path.join('emoji', name) for name in os.listdir('emoji') if name.endswith('.png')][:count]
        images = np.array([load_image(path, 64) for path in image_paths])
        noise = np.random.uniform(size=[count, self.noise_size])
        use_real_image = np.full([count], 0)
        images = self.session.run(self.disc_input, feed_dict={
            self.real_image_input: images,
            self.use_real_image: use_real_image,
            self.disc_dropout_keep_prob: 1,
            self.noise_input: noise
        })
        return images

n = GAN(dir_path='model/gan5')

def train():
    image_paths = [os.path.join('emoji', name) for name in os.listdir('emoji') if name.endswith('.png')]
    images = np.array([load_image(path, 64) for path in image_paths])
    train_batcher = batch_generator(images, images, size=64, random=True)
    test_batcher = batch_generator(images, images, size=64, random=True)
    n.training_loop(train_batcher, test_batcher)

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
