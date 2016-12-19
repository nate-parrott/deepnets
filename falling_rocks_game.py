import random
import numpy as np
from copy import deepcopy

class Game(object):
    screen_size = (16, 16)
    bowl_size = 3
    rock_prob = 0.15
    
    def __init__(self):
        pass
    
    def initial_state(self):
        return {
            "rocks": [self.create_rock() for _ in range(1)],
            "player_x": random.randint(0, self.screen_size[0])
        }
    
    def actions(self):
        return ['none', 'left', 'right']
    
    def iterate(self, state, action):
        newstate = deepcopy(state)
        # move rocks down:
        for rock in newstate['rocks']:
            rock['y'] += 1
        newstate['rocks'] = [r for r in newstate['rocks'] if r['y'] < self.screen_size[1]]
        # create new rock, maybe?
        if random.random() < self.rock_prob:
            newstate['rocks'].append(self.create_rock())
        # move the player:
        if action == 'right':
            newstate['player_x'] = min(self.screen_size[0] - 1, newstate['player_x'] + 1)
        elif action == 'left':
            newstate['player_x'] = max(0, newstate['player_x'] - 1)
        # calculate reward:
        reward = 0.5
        for rock in newstate['rocks']:
            if rock['y'] == self.screen_size[1] - 1:
                dist = abs(rock['x'] - newstate['player_x'])
                if dist <= (self.bowl_size - 1)/2:
                    reward *= 2
                else:
                    reward = 0
        
        return newstate, reward
    
    def render(self, state):
        n = np.zeros(self.screen_size)
        for rock in state['rocks']:
            n[(rock['x'], rock['y'])] = 1
        bowl_start = state['player_x'] - (self.bowl_size-1)/2
        for x in range(bowl_start, bowl_start + self.bowl_size):
            if x >= 0 and x < self.screen_size[0]:
                n[(x, self.screen_size[1]-1)] = 1
        return n
    
    def print_render(self, render):
        print '-' * render.shape[0]
        for y in xrange(render.shape[1]):
            row = [('X' if render[(x,y)] > 0 else ' ') for x in xrange(render.shape[0])]
            print ''.join(row)
        print '=' * render.shape[0]
        print ''
    
    # HELPERS:
    def create_rock(self):
        return {"x": random.randint(0, self.screen_size[0] - 1), "y": 0}

class EasyGame(Game):
    def create_rock(self):
        return {"x": 0, "y": 0}

if __name__ == '__main__':
    g = EasyGame()
    s = g.initial_state()
    while True:
        g.print_render(g.render(s))
        action = raw_input('/'.join(g.actions()) + ' > ')
        if action not in g.actions():
            actions = g.actions()[0]
        s, reward = g.iterate(s, action)
        print 'REWARD:', reward
