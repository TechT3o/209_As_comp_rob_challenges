# Contains necessary information on which the numberline problem is modeled

import math
import random as rng

class NumberlineSystem:

    def __init__(self):
        self.v = 0
        self.y = 0

        self.time_index = 0
        self.horizon = 0
        self.gamma = 1

        self.applied_force = [-1, 0, 1]
        self.A = 1
        self.pw = 1
        self.pc = 1
        self.y_max = 20
        self.v_max = 20
        self.y_states = range(-self.y_max, self.y_max, 1)
        self.v_states = range(-self.v_max, self.v_max, 1)
        self.constant_force = int(self.A * math.sin(2 * math.pi * self.y / self.y_max))

        self.value = [0]

    def speed_wobble(self):
        p = rng.uniform(0,1)
        distribution_threshold = (self.v / self.v_max) * self.pw
        if 0 < p <= distribution_threshold/2:
            return 1
        if distribution_threshold/2 < p <= distribution_threshold:
            return -1
        if distribution_threshold < p <=1:
            return 0
        else:
            print("Error")


    def crashing_prob(self):
        return self.pc * self.v / self.v_max

    def value_iteration(self):
        # TODO: adjust states and actions, add p function
        v = [0]
        i = 0
        while i != self.horizon:
            new_v = 0
            for j in range(len(self.states)):
                new_v += p(self.states[j], self.actions[j], self.states[j+1]) * (r(self.states[j]) + (self.gamma*v[-1]))
                v.append(new_v)
            i += 1
        return v