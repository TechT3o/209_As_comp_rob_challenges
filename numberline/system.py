# Contains necessary information on which the numberline problem is modeled

import math
import random as rng
import itertools
import numpy as np
class NumberlineSystem:

    def __init__(self):
        self.v = 0
        self.y = 0

        self.time_index = 0
        self.horizon = 150
        self.gamma = 0.8
        self.particle_mass = 1

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

    def reward_old(self, x, u=None):
        if x == (0, 0): # at rest at origin
            return 1
        else:
            return 0
    def speed_wobble(self):
        p = rng.uniform(0, 1)
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
        return self.pc * abs(current_v) / self.v_max

    def noise_prob(self, current_v):
        return (abs(current_v) / self.v_max) * self.pw

    def find_constant_force(self, current_pos):
        return int(self.A * math.sin(2 * math.pi * current_pos / self.y_max))

    def value_iteration(self):
        v = [0 for _ in range(len(self.state_space))]
        pi = [None for _ in range(len(self.state_space))]
        for i in range(self.horizon):
            max_diff = 0
            V_new = [0 for _ in range(len(self.state_space))]
            for state in self.state_space:
                max_val = [] # keep track of best value
                # if self.check_laws_of_motion(state):
                #     continue
                for action in self.applied_force:
                    val = self.reward(state)
                    for next_state in self.state_space:
                        trans_prob = self.get_transition_prob(action, state, next_state)
                        increment =  trans_prob * ( self.gamma * v[self.state_space.index(next_state)])
                        # if increment > 0:
                        #     print(increment)
                        #     print(state, next_state, trans_prob)
                        val += increment
                        #print(f'next value for state {state}, next state {next_state}, action {action} is {val}')

                    max_val.append(val) # update max

                    # if V_new[self.state_space.index(state)] < val:
                    #     pi[self.state_space.index(state)] = self.applied_force[self.applied_force.index(action)]

                pi[self.state_space.index(state)] = self.applied_force[max_val.index(max(max_val))]
                V_new[self.state_space.index(state)] = max(max_val)

                max_diff = max(max_diff, abs(v[self.state_space.index(state)] - V_new[self.state_space.index(state)]))
            #print(v)
            v = V_new

        return v, pi

    def value_iteration_new(self):
        v = [0 for _ in range(len(self.state_space))]
        pi = [None for _ in range(len(self.state_space))]
        for i in range(self.horizon):
            max_diff = 0
            V_new = [0 for _ in range(len(self.state_space))]
            for state in self.state_space:
                max_val = [] # keep track of best value
                for action in self.applied_force:
                    val = self.reward(state)
                    for next_state in self.state_space:
                        trans_prob = self.get_transition_prob(action, state, next_state)
                        # if state == (-5,-5):
                        #     print(trans_prob, action, next_state)
                        increment = trans_prob * (self.gamma * v[self.state_space.index(next_state)])
                        val += increment

                    max_val.append(val) # update max

                pi[self.state_space.index(state)] = self.applied_force[max_val.index(max(max_val))]
                V_new[self.state_space.index(state)] = max(max_val)

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

        if next_pos-curr_pos != curr_vel and abs(curr_pos + curr_vel) <= self.y_max:
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

    def check_laws_of_motion(self, curr_state):
        curr_pos = curr_state[0]
        curr_vel = curr_state[1]

        if abs(curr_pos + curr_vel) > self.y_max:
            return True
        else:
            return False

    def policy_evalutation(self, v, pi):
        diff = 0
        threshold = 10 ** -10
        while diff < threshold:
            v_new = [0 for _ in range(len(self.state_space))]
            for state in self.state_space:
                val = 0
                for next_state in self.state_space:
                    trans_prob = self.get_transition_prob(pi[self.state_space.index(state)], state, next_state)
                    increment = trans_prob * (self.reward(state) + self.gamma * v[self.state_space.index(next_state)])
                    val += increment
                v_new[self.state_space.index(state)] = val

                diff = max(diff, abs(v[self.state_space.index(state)] - v_new[self.state_space.index(state)]))
        return v_new

    def policy_iteration(self, v, pi):
        print('iterated')
        new_v = self.policy_evalutation(v, pi)
        print(new_v)
        # policy improvement
        policy_condition = True
        old_policy = pi
        for state in self.state_space:

            max_val = []
            for action in self.applied_force:
                val = self.reward(state)
                for next_state in self.state_space:
                    trans_prob = self.get_transition_prob(action, state, next_state)
                    increment = trans_prob * (self.gamma * new_v[self.state_space.index(next_state)])
                    val += increment

                max_val.append(val)  # update max
            pi[self.state_space.index(state)] = self.applied_force[max_val.index(max(max_val))]
        if old_policy != pi:
            policy_condition = False
        if policy_condition:
            return v, pi
        else:
            self.policy_iteration(new_v, pi)

    def do_policy_iteration(self):
        v = [0 for _ in range(len(self.state_space))]
        pi = [rng.randint(-1, 1) for _ in range(len(self.state_space))]
        optimal_v, optimal_pi = self.policy_iteration(v, pi)
        return optimal_v, optimal_pi

    def pick_random_state(self):
        state = (rng.uniform(-self.y_max, self.y_max), rng.uniform(-self.v_max, self.v_max))
        return state

    def check_if_connected(self, state_1, state_2):
        pos_1 = state_1[0]
        pos_2 = state_2[0]
        vel_1 = state_1[1]
        vel_2 = state_2[1]
        curr_constant_force_1_2 = self.find_constant_force(pos_1)
        curr_constant_force_2_1 = self.find_constant_force(pos_2)
        force_required_for_v_change_1_2 = (vel_2 - vel_1) - (1 / self.particle_mass) * curr_constant_force_1_2
        force_required_for_v_change_2_1 = (vel_1 - vel_2) - (1 / self.particle_mass) * curr_constant_force_2_1
        connected = False
        direction = None
        force = None

        if not (pos_2-pos_1 != vel_1 and abs(pos_1 + vel_1) <= self.y_max) and (-1 <= force_required_for_v_change_1_2 <= 1):
            connected = True
            force = force_required_for_v_change_1_2
            direction = 1

        if not (pos_1-pos_2 != vel_2 and abs(pos_2 + vel_2) <= self.y_max) and (-1 <= force_required_for_v_change_2_1 <= 1):
            connected = True
            force = force_required_for_v_change_2_1
            direction = -1

        return connected, force, direction

if __name__ == "__main__":
    nls = NumberlineSystem()
    # v,pi = nls.value_iteration()
    # # v,pi = nls.do_policy_iteration()
    # for i in range(len(nls.state_space)):
    #     print(f' policy is {pi[i]} for state {nls.state_space[i]} with value {v[i]}')
    # arr = np.array([i for i in pi if i != None])
    # print(arr == -arr[::-1])
    state_list = [nls.pick_random_state()]
    for i in range(50):
        state_list.append(nls.pick_random_state())
        for state in state_list[:-1]:
            connected, force, direction = nls.check_if_connected(state, state_list[-1])
            print(i, connected, force, direction)
            # TODO: Create adjacency matrix calculation function and implement an optimal path solving algorithm
