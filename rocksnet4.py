from net import Net
import numpy as np
import tensorflow as tf
from falling_rocks_game import Game, EasyGame, MiddleGame, BigGame
from move_game import MoveGame
from util import weight_var
from tensorflow.contrib.layers.python.layers import batch_norm
import random
from tensorflow.contrib.layers.python.layers import batch_norm
# https://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/
# https://github.com/DanielSlater/PyGamePlayer/blob/master/examples/deep_q_pong_player.py
import sys

show = 'show' in sys.argv

game = Game(negative_rewards=False)

class Player(Net):
    def setup(self):
        game_screens = tf.placeholder(tf.float32, [None, game.screen_size[0], game.screen_size[1]], name='game_screens')
        target_actions = tf.placeholder(tf.int64, [None], name='target_actions')
        rewards = tf.placeholder(tf.float32, [None], name='rewards')
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        
        screens = tf.reshape(game_screens, [-1, game.screen_size[0] * game.screen_size[1]])
        h0 = create_dense(screens, game.screen_size[0] * game.screen_size[1], 64)
        # h1 = create_dense(h0, 64, 32)
        h2 = create_dense(h0, 64, 32)
        actions = create_dense(h2, 32, len(game.actions()))
        
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(actions, target_actions) * rewards)
        
        self.game_screens = game_screens
        self.target_actions = target_actions
        self.rewards = rewards
        self.dropout_keep_prob = dropout_keep_prob
        
        self.actions = tf.argmax(tf.nn.softmax(actions), 1)
        self.loss = loss
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    def train(self, training_tuple):
        # experiences: [(screen, action_idx, reward)]
        screens, action_indices, rewards = training_tuple
        feed_dict = {self.dropout_keep_prob: 0.7, self.game_screens: screens, self.target_actions: action_indices, self.rewards: rewards}
        loss, _ = self.session.run([self.loss, self.train_step], feed_dict=feed_dict)
        # print 'Loss:', loss
    
    def predict_actions(self, screens):
        return self.session.run(self.actions, feed_dict={self.dropout_keep_prob: 1, self.game_screens: screens}) 

def create_conv(input, in_channels, out_channels, weight_set=[]):
    patch_size = 3
    w = weight_var([patch_size, patch_size, in_channels, out_channels], stddev=0.1)
    b = weight_var([out_channels], stddev=0)
    conv = tf.nn.conv2d(input, w, strides=[1,2,2,1], padding='SAME')
    activation = tf.nn.relu(conv + b)
    weight_set.append(w)
    weight_set.append(b)
    return activation

def create_dense(input, in_size, out_size, weight_set=[], relu=True, no_bias=False):
    # input_dropped = tf.nn.dropout(input, dropout_keep_prob)
    w = weight_var([in_size, out_size], stddev=0.01)
    weight_set.append(w)
    b = weight_var([out_size], stddev=0)
    if not no_bias:
        weight_set.append(b)
    x = tf.matmul(input, w)
    return tf.nn.relu(x + b) if relu else x + b

def flatten(lists):
    return [x for l in lists for x in l]

def l2_loss(weights, l2=0.001):
    loss = tf.nn.l2_loss(weights[0])
    for w in weights[1:]:
        loss += tf.nn.l2_loss(w)
    return 0.01 * loss

def avg(values):
    return sum(values) * 1.0 / len(values)

def backfill_rewards(experiences, reward_decay=0.8):
    def frontfill_rewards(experiences):
        trailing_reward = 0
        for (screen, action, reward) in experiences:
            trailing_reward = reward + trailing_reward * reward_decay
            yield (screen, action, trailing_reward)
        
    l = list(frontfill_rewards(reversed(experiences)))
    l.reverse()
    return l

training_tuples = []

def train(player_net, explore_prob=0.1, evaluate=False):
    
    n_games = 128
    if show: n_games = 1
    n_steps = 32
    all_actions = game.actions()
    all_action_indices = range(len(all_actions))
    states = [game.initial_state() for _ in xrange(n_games)]
    screens = np.stack([game.render(state) for state in states])
    experiences_by_game = [[] for _ in xrange(n_games)]
        
    total_reward = 0
    
    for step in xrange(n_steps):
        experiences_for_step = []
        
        actions = player_net.predict_actions(screens)
        actions = [(random.choice(all_action_indices) if random.random() < explore_prob else best_action) for best_action in actions]
        
        new_states = []
        
        for idx, (state, action_idx, screen) in enumerate(zip(states, actions, screens)):
            new_state, reward = game.iterate(state, all_actions[action_idx])
            total_reward += reward
            
            if show:
                print 'step:', step
                game.print_render(game.render(state))
                print 'action:', all_actions[action_idx]
                print 'reward:', reward
            
            new_states.append(new_state)
            experiences_by_game[idx].append((screen, action_idx, reward))
                
        states = new_states
        screens = np.stack([game.render(state) for state in states])
        
    experiences = flatten([backfill_rewards(exp) for exp in experiences_by_game])
    experiences = [exp for exp in experiences if exp[2] > 0.2]
    
    screens = np.stack((e[0] for e in experiences))
    action_indices = np.stack((e[1] for e in experiences))
    rewards = np.stack((e[2] for e in experiences))
    
    global training_tuples
    training_tuples = [(screens, action_indices, rewards)] + training_tuples    
    if len(training_tuples) > 20: training_tuples = training_tuples[20:]
    
    if not evaluate:
        for _ in xrange(10):
            player_net.train(random.choice(training_tuples))
    
    return total_reward / float(n_games)

def run():
    player_net = Player(dir_path='model/rocksnet4')
    step = 0
    while True:
        step += 1
        explore_prob = 0 if show else max(0.05, 1.0 - step / 5000.0)
        if step % 5 == 0:
            # this step, let's evaluate and save
            print "Average reward per game:", train(player_net, 0, evaluate=True)
            player_net.save(step)
            print "Saved"
            print "Explore prob is {0} at step {1}".format(explore_prob, step)
        else:
            new_experiences = train(player_net, explore_prob)

if __name__ == '__main__':
    run()
