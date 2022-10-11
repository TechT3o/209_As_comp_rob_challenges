from itertools import combinations

class GridworldSystem:
    action_space: list
    initial_state: list
    states: list
    probability_error: float

    def __init__(self):
        self.action_space = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]
        self.initial_state = [0, 0]
        self.states = [[0,0],]
        self.probability_error = 0.1
        self.forbidden_states = []

    def transition_function(self):
        next_state = 1

    def check_if_forbidden(self, next_state):
        if next_state in self.forbidden_states:
            return True