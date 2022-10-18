# Contains necessary information on which the numberline problem is modeled

import math
import random as rng
import itertools

class NumberlineSystem:

    def __init__(self):
        self.v = 0
        self.y = 0

        self.time_index = 0
        self.horizon = 100
        self.gamma = 1

        self.applied_force = [-1, 0, 1]
        self.A = 1
        self.pw = 1
        self.pc = 1
        self.y_max = 5
        self.v_max = 5
        self.state_space = [[(y, v) for v in range(-self.v_max, self.v_max + 1)] for y in range(-self.y_max, self.y_max + 1)]
        self.state_space = list(itertools.chain(*self.state_space)) # flatten list

        self.constant_force = int(self.A * math.sin(2 * math.pi * self.y / self.y_max))

        self.value = [0]

    def reward(self, x, u=None):
        if x == (0, 0): # at rest at origin
            return 10
        elif x[1] != 0: # fuel cost
            return -1
        else:
            return 0

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
            raise Exception(f"p value of {p} caused error in speed_wobble()")

    def crashing_prob(self, current_v):
        return self.pc * current_v / self.v_max

    def noise_prob(self, current_v):
        return (current_v / self.v_max) * self.pw

    def find_constant_force(self, current_pos):
        return int(self.A * math.sin(2 * math.pi * current_pos / self.y_max))

    def value_iteration(self):
        v = [0 for _ in range(len(self.state_space))]
        pi = [None for _ in range(len(self.state_space))]
        for i in range(self.horizon):
            max_diff = 0
            V_new = [0 for _ in range(len(self.state_space))]
            for state in self.state_space:
                max_val = 0 # keep track of best value
                for action in self.applied_force:
                    val = self.reward(state)
                    for next_state in self.state_space:
                        print(state, next_state, self.get_transition_prob(action, state, next_state))
                        val += self.get_transition_prob(action, state, next_state) * (self.gamma * v[self.state_space.index(next_state)])

                    max_val = max(max_val, val) # update max

                    if V_new[self.state_space.index(state)] < val:
                        pi[self.state_space.index(state)] = self.applied_force[self.applied_force.index(action)]

                V_new[self.state_space.index(state)] = max_val

                max_diff = max(max_diff, abs(v[self.state_space.index(state)] - V_new[self.state_space.index(state)]))

            v = V_new

        return v, pi

    def get_transition_prob(self, input, curr_state, next_state):
        curr_pos = curr_state[0]
        next_pos = next_state[0]
        curr_vel = curr_state[1]
        next_vel = next_state[1]
        curr_constant_force = self.find_constant_force(curr_pos)
        velocity_difference = next_vel - curr_vel - curr_constant_force

        if next_pos-curr_pos != curr_vel:
            return 0

        if input == 0:
            if velocity_difference == 1:
                return self.noise_prob(curr_vel)/2 * (1- self.crashing_prob(curr_vel))
            if velocity_difference == 0:
                return (1 - self.noise_prob(curr_vel))*(1- self.crashing_prob(curr_vel))
            if velocity_difference == -1:
                return self.noise_prob(curr_vel) / 2 * (1 - self.crashing_prob(curr_vel))
            if velocity_difference == - curr_vel - curr_constant_force:
                return self.crashing_prob(curr_vel)
            else:
                return 0

        if input == 1:
            if velocity_difference == 2:
                return self.noise_prob(curr_vel)/2 * (1- self.crashing_prob(curr_vel))
            if velocity_difference == 1:
                return (1 - self.noise_prob(curr_vel))*(1- self.crashing_prob(curr_vel))
            if velocity_difference == 0:
                return self.noise_prob(curr_vel) / 2 * (1 - self.crashing_prob(curr_vel))
            if velocity_difference == - curr_vel - curr_constant_force:
                return self.crashing_prob(curr_vel)
            else:
                return 0

        if input == -1:
            if velocity_difference == 0:
                return self.noise_prob(curr_vel)/2 * (1- self.crashing_prob(curr_vel))
            if velocity_difference == -1:
                return (1- self.noise_prob(curr_vel))*(1- self.crashing_prob(curr_vel))
            if velocity_difference == -2:
                return self.noise_prob(curr_vel) / 2 * (1 - self.crashing_prob(curr_vel))
            if velocity_difference == - curr_vel - curr_constant_force:
                return self.crashing_prob(curr_vel)
            else:
                return 0

if __name__ == "__main__":
    nls = NumberlineSystem()
    print(nls.value_iteration())
