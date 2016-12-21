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
discount = 0.99

game = BigGame()

class Player(Net):
    def setup(self):
        game_screens = tf.placeholder(tf.float32, [None, game.screen_size[0], game.screen_size[1]], name='game_screens')
        target_outputs = tf.placeholder(tf.float32, [None, len(game.actions())], name='reward')
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        
        patch_size = 3
        weights = []
        
        def create_conv(input, in_channels, out_channels, weight_set=[]):
            w = weight_var([patch_size, patch_size, in_channels, out_channels], stddev=0.01)
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
        
        conv = False
        
        if conv:
            screens = tf.reshape(game_screens, [-1, game.screen_size[0], game.screen_size[1], 1]) # 16 x 16 x 1
            screens = create_conv(screens, 1, 8, weight_set=weights) # 8 x 8 x 8
            screens = create_conv(screens, 8, 8, weight_set=weights) # 4 x 4 x 8
            screens_size = game.screen_size[0] / 4 * game.screen_size[1] / 4 * 8 # 4 x 4 x 8 = 128
            screens = tf.reshape(screens, [-1, screens_size])
            screens = batch_norm(screens)
            # h0 = create_dense(screens, screens_size, 32, weight_set=weights)
            # h1 = create_dense(h0, 32, len(game.actions()))
            h1 = create_dense(screens, screens_size, len(game.actions()), weight_set=weights)
        else:
            h0 = create_dense(tf.reshape(game_screens, [-1, game.screen_size[0] * game.screen_size[1]]), game.screen_size[0] * game.screen_size[1], 64, weight_set=weights)
            h0 = create_dense(h0, 64, 32, weight_set=weights)
            # h0 = batch_norm(h0)
            h1 = create_dense(h0, 32, len(game.actions()), weight_set=weights)
            # h1 = batch_norm(h1)
        
        loss = tf.reduce_mean(tf.square(target_outputs - h1)) # + l2_loss(weights)
        optimizer = tf.train.RMSPropOptimizer(0.0000001, momentum=0.1, epsilon=0.01)
        train_step = optimizer.minimize(loss)
        
        tf.scalar_summary('loss', loss)
        self.summary_op = tf.merge_all_summaries()
        
        # inputs:
        self.game_screens = game_screens
        self.dropout_keep_prob = dropout_keep_prob
        self.target_outputs = target_outputs
        
        # outputs:
        self.q_vals = h1
        self.chosen_actions = tf.argmax(h1, 1)
        self.train_step = train_step
        self.loss = loss
        
        self.past_experiences = []
    
    def q_vals_for_screens(self, screens):
        return self.session.run(self.q_vals, feed_dict={self.dropout_keep_prob: 1, self.game_screens: screens})
        
    def train(self, experiences):
        # experiences: [(screen, action_idx, new_screen, reward)]
        all_actions = game.actions()
        screens = np.stack(e[0] for e in experiences)
        target_outputs = self.q_vals_for_screens(screens) # initialize to predicted q-values
        # print target_outputs[0]
        # print 'q vals for all screens during training:'
        if 'print_q' in sys.argv:
            print np.argmax(target_outputs, axis=1)
            print target_outputs[0]
        # print 'BEFORE'
        # print target_outputs
        
        new_state_qs = self.q_vals_for_screens(np.stack(e[2] for e in experiences))
        for i, (old_screen, action_idx, new_screen, reward) in enumerate(experiences):
            target_outputs[i][action_idx] = reward + discount * new_state_qs[i].max()
        # print 'TARGETS'
        # print target_outputs
        feed_dict = {self.dropout_keep_prob: 0.7, self.game_screens: screens, self.target_outputs: target_outputs}
        _, loss, _ = self.session.run([self.train_step, self.loss, self.summary_op], feed_dict=feed_dict)
        # print 'AFTER'
        # print self.q_vals_for_screens(screens) # initialize to predicted q-values
        # print "Loss:", loss

def flatten(lists):
    return [x for l in lists for x in l]

def l2_loss(weights, l2=0.001):
    loss = tf.nn.l2_loss(weights[0])
    for w in weights[1:]:
        loss += tf.nn.l2_loss(w)
    return 0.01 * loss

def avg(values):
    return sum(values) * 1.0 / len(values)

def train(player_net, explore_prob=0.1, evaluate=False):
    experiences = []
    
    n_games = 1 if show else 32
    if evaluate: n_games = 128
    n_steps = 64
    all_actions = game.actions()
    all_action_indices = range(len(all_actions))
    states = [game.initial_state() for _ in xrange(n_games)]
    screens = np.stack([game.render(state) for state in states])
    
    for step in xrange(n_steps):
        q_vals = player_net.q_vals_for_screens(screens)
        # print 'q vals at step {0}: {1}'.format(step, np.argmax(q_vals, axis=1))
        #print q_vals
        noisy_q_vals = q_vals # + (np.random.rand(q_vals.shape[0], q_vals.shape[1]) - 0.5) * 0.001
        actions = [(random.choice(all_action_indices) if random.random() < explore_prob else best_action) for best_action in np.argmax(noisy_q_vals, axis=1)]
        # actions = [1 for _ in np.argmax(noisy_q_vals, axis=1)]
        
        if show:
            print 'q vals:', q_vals
            print actions
        
        new_states = []
        rewards = []
        # print len(states), len(actions)
        for state, action_idx in zip(states, actions):
            new_state, reward = game.iterate(state, all_actions[action_idx])
            
            if show:
                print 'step:', step
                game.print_render(game.render(state))
                print 'action:', all_actions[action_idx]
                print 'reward:', reward
            
            new_states.append(new_state)
            rewards.append(reward)
                
        new_screens = np.stack([game.render(state) for state in new_states])
        experiences += list(zip(screens, actions, new_screens, rewards))
        
        states = new_states
        screens = new_screens
    
    if not evaluate:
        player_net.train(experiences)
        # player_net.past_experiences += experiences
        # player_net.past_experiences = player_net.past_experiences[:min(len(player_net.past_experiences), 8096 * 32)]
        # if len(player_net.past_experiences) < 8192:
        #     player_net.train(player_net.past_experiences)
        # else:
        #     idx = random.randint(0, len(player_net.past_experiences) - 8192)
        #     player_net.train(player_net.past_experiences[idx:idx+8192])
    return avg([e[3] for e in experiences]) * n_steps

def run():
    player_net = Player(dir_path='model/rocks-big-fc')
    step = 0
    while True:
        step += 1
        explore_prob = 0 if show else max(0.05, 1.0 - step / 5000.0)
        if step % 20 == 0:
            # this step, let's evaluate and save
            print "Average reward per game:", train(player_net, 0, evaluate=True)
            player_net.save(step)
            print "Saved"
            print "Explore prob is {0} at step {1}".format(explore_prob, step)
        else:
            new_experiences = train(player_net, explore_prob)

if __name__ == '__main__':
    run()
