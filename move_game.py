import random
import numpy as np
from copy import deepcopy

class MoveGame(object):
    screen_size = (6, 6)
    
    def __init__(self):
        pass
    
    def initial_state(self):
        return {
            "x": random.randint(0, self.screen_size[0]-1),
            "y": random.randint(0, self.screen_size[1]-1)
        }
    
    def actions(self):
        return ['left', 'right', 'up', 'down', 'none']
    
    def iterate(self, state, action):
        newstate = deepcopy(state)
        if action == 'left':
            newstate['x'] = max(0, newstate['x']-1)
        elif action == 'right':
            newstate['x'] = min(self.screen_size[0]-1, newstate['x'] + 1)
        elif action == 'down':
            newstate['y'] = max(0, newstate['y']-1)
        elif action == 'up':
            newstate['y'] = min(self.screen_size[1]-1, newstate['y'] + 1)
        reward = 1 if newstate['x'] == self.screen_size[0]/2 and newstate['y'] == self.screen_size[1]/2 else 0
        return newstate, reward
    
    def render(self, state):
        n = np.zeros(self.screen_size)
        n[(state['x'], state['y'])] = 1
        return n
    
    def print_render(self, render):
        print '-' * render.shape[0]
        for y in xrange(render.shape[1]):
            row = [('X' if render[(x,y)] > 0 else ' ') for x in xrange(render.shape[0])]
            print ''.join(row)
        print '=' * render.shape[0]
        print ''

if __name__ == '__main__':
    g = MoveGame()
    s = g.initial_state()
    while True:
        g.print_render(g.render(s))
        action = raw_input('/'.join(g.actions()) + ' > ')
        if action not in g.actions():
            actions = g.actions()[0]
        s, reward = g.iterate(s, action)
        print 'REWARD:', reward
