from net import Net
import numpy as np
import tensorflow as tf
from falling_rocks_game import Game
from util import weight_var
from tensorflow.contrib.layers.python.layers import batch_norm
import random

noise_size = 4

game = Game()

class Player(Net):
    def setup(self):
        noise = tf.placeholder(tf.float32, [None, noise_size], name='noise')
        game_screens = tf.placeholder(tf.float32, [None, game.screen_size[0], game.screen_size[1]], name='game_screens')
        target_outputs = tf.placeholder(tf.float32, [None, len(game.actions())], name='reward')
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        
        patch_size = 3
        
        def create_conv_with_pooling(input, in_channels, out_channels, weight_set=[]):
            w = weight_var([patch_size, patch_size, in_channels, out_channels])
            b = weight_var([out_channels])
            conv = tf.nn.conv2d(input, w, strides=[1,1,1,1], padding='SAME')
            activation = tf.nn.relu(conv + b)
            weight_set.append(w)
            weight_set.append(b)
            pooled = tf.nn.max_pool(activation, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            return pooled
        
        def create_conv(input, in_channels, out_channels, weight_set=[]):
            w = weight_var([patch_size, patch_size, in_channels, out_channels])
            b = weight_var([out_channels])
            conv = tf.nn.conv2d(input, w, strides=[1,2,2,1], padding='SAME')
            activation = tf.nn.relu(conv + b)
            weight_set.append(w)
            weight_set.append(b)
            return activation
        
        def create_dense(input, in_size, out_size, weight_set=[], relu=True, no_bias=False):
            input_dropped = tf.nn.dropout(input, dropout_keep_prob)
            w = weight_var([in_size, out_size])
            weight_set.append(w)
            b = weight_var([out_size], init_zero=no_bias)
            if not no_bias:
                weight_set.append(b)
            x = tf.matmul(input_dropped, w)
            # x = batch_norm(x)
            return tf.nn.relu(x + b) if relu else x + b
        
        screens = tf.reshape(game_screens, [-1, game.screen_size[0], game.screen_size[1], 1]) # 16 x 16 x 1
        screens = create_conv(screens, 1, 4) # 8 x 8 x 4
        screens = create_conv(screens, 4, 8) # 4 x 4 x 8
        out_size = game.screen_size[0] / 4 * game.screen_size[1] / 4 * 8
        screens = tf.reshape(screens, [-1, out_size])
        
        h0 = tf.concat(1, [screens, noise])
        h1 = create_dense(h0, out_size + noise_size, len(game.actions()))
        # softmax, maybe?
        
        # target_output = tf.Print(target_output, [tf.shape(target_output)], message="target output shape:")
        # action = tf.Print(action, [tf.shape(action)], message="action shape:")
        # h1 = tf.Print(h1, [tf.shape(h1)], message="h1 shape:")
        
        
        loss = tf.reduce_sum(tf.pow(target_outputs - h1, 2))
        optimizer = tf.train.AdamOptimizer(0.0001)
        train_step = optimizer.minimize(loss)
        
        # inputs:
        self.noise = noise
        self.game_screens = game_screens
        self.dropout_keep_prob = dropout_keep_prob
        self.target_outputs = target_outputs
        
        # outputs:
        self.action_probs = h1
        self.chosen_actions = tf.argmax(h1, 1)
        self.train_step = train_step
        self.loss = loss
    
    def train(self, game_states, rewards, actions):
        
        # flatten all the states -- don't need to group games by state:
        rewards = flatten(backfill_rewards(rewards))
        game_states = flatten(game_states)
        actions = flatten(actions)
                
        # turn game states into vectors:
        game_states_np = np.array([game.render(state) for state in game_states])
        
        noise = np.random.uniform(size=[len(game_states), noise_size])
        
        # feed-forward to find action probabilities:
        action_probs = self.session.run(self.action_probs, feed_dict={
            self.noise: noise,
            self.game_screens: game_states_np,
            self.dropout_keep_prob: 1
        })
        
        # make one-hot action vectors:
        action_names = game.actions()
        target_outputs = np.zeros((len(actions), len(action_names)))
        # actions_np.fill(-0.1)
        for i, action in enumerate(actions):
            target_outputs[action_names.index(action)] = rewards[i]
                
        feed = {
            self.noise: noise,
            self.game_screens: game_states_np,
            self.target_outputs: target_outputs,
            self.dropout_keep_prob: 0.5
        }
        _, loss = self.session.run([self.train_step, self.loss], feed_dict=feed)
        print "Training loss:", loss
    
    def get_actions(self, game_states):
        noise = np.random.uniform(size=[len(game_states), noise_size])
        game_states_np = np.array([game.render(state) for state in game_states])
        
        feed = {
            self.noise: noise,
            self.game_screens: game_states_np,
            self.dropout_keep_prob: 1
        }
        actions = self.session.run(self.chosen_actions, feed_dict=feed)
        names = game.actions()
        return [names[i] for i in actions]

def flatten(lists):
    return [x for l in lists for x in l]

def avg(values):
    return sum(values) * 1.0 / len(values)

def backfill_rewards(rewards_by_step):
    outputs = [ [0 for _ in xrange(len(rewards_by_step[0]))] for _ in xrange(len(rewards_by_step)) ]
    back_dist = 5
    back_coefficients = [(1 - i * 1.0 / back_dist) ** 2 for i in xrange(back_dist)]
    for step, rewards in enumerate(rewards_by_step):
        for game, reward in enumerate(rewards):
            if reward != 0:
                for back_step, k in enumerate(back_coefficients):
                    if step - back_step >= 0:
                        outputs[step - back_step][game] += reward * k
                    else: break
    return outputs

def train_step(player_net, explore_prob=0.1):
    n_games = 64
    n_steps = 16
    
    all_actions = game.actions()
    
    game_states = [[game.initial_state() for _ in xrange(n_games)]]
    rewards = [[0 for _ in xrange(n_games)]]
    actions = [[random.choice(all_actions) for _ in xrange(n_games)]]
    for step in xrange(n_steps):
        actions_at_step = player_net.get_actions(game_states[-1])
        actions_at_step = [random.choice(all_actions) if random.random() < explore_prob else action for action in actions_at_step]
        rewards_at_step = []
        game_states_at_step = []
        for i in xrange(n_games):
            game_state, reward = game.iterate(game_states[-1][i], actions_at_step[i])
            game_states_at_step.append(game_state)
            rewards_at_step.append(reward)
        game_states.append(game_states_at_step)
        rewards.append(rewards_at_step)
        actions.append(actions_at_step)
        # print explore_prob
        # print actions_at_step
    
    player_net.train(game_states, rewards, actions)
    
    all_rewards = [r for rewards_at_step in rewards for r in rewards_at_step]
    print "Average reward:", avg(all_rewards)

def run():
    n = Player()
    step = 0
    while True:
        explore_prob = max(0.2, 1.0 - step / 50000.0)
        train_step(n, explore_prob)
        step += 1

if __name__ == '__main__':
    run()
