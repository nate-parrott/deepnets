import tensorflow as tf
import numpy as np
from net import Net, batch_generator
from simple import *
from util import weight_var, create_fc, create_conv, create_deconv
from tensorflow.contrib.layers.python.layers import batch_norm

TEXT_LEN = 16
NOISE_SIZE = 16

ALPHABET = 'abcdefghijklmnopqrstuvwxyz '
# zero-vec represents the end of the sentence

class TextNet(Net):
    def setup(self):
        pass
    
    def build_discriminator(self, strings, input):
        # takes matrix of strings
        # returns a tuple:
        #   - 1d tensor tensor that's the probability of each string being real
        #   - list of weights used
        pass
    
    def build_generator(self, random_noise):
        # takes matrix of noise vectors
        # returns matrix of strings

def text_to_vec(text):
    indices = [ALPHABET.index(c) + 1 for c in text.lower() if c in ALPHABET]
    return np.array(indices + [0] * (TEXT_LEN - len(indices)), dtype=int)    
