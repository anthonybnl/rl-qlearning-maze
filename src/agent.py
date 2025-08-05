import random


class Agent:
    def __init__(
        self, initial_epsilon: float, epsilon_decay: float, final_epsilon: float
    ):

        self.qvalues: dict[int, dict[int, float]] = {}
        self.actions_par_etat = {}

        # parameters

        self.lr = 0.99  # on conserve 90% de la valeur actuellement connue
        self.discount_factor = 0.9  # How much we care about future rewards : 90%

        self.initial_epsilon = initial_epsilon
        self.epsilon = self.initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def choisir_action(self, state, actions_possibles):

        # on mets à jour la liste des actions possibles
        if state not in self.actions_par_etat:
            self.actions_par_etat[state] = actions_possibles
            self.qvalues[state] = {}
            for action in actions_possibles:
                self.qvalues[state][action] = 0.0

        if random.random() < self.epsilon:
            # action aléatoire
            action = actions_possibles[random.randrange(len(actions_possibles))]
        else:
            actions_for_this_state = list(self.qvalues[state].keys())
            best_action = actions_for_this_state[0]
            max_q = self.qvalues[state][best_action]
            for action in actions_for_this_state:
                q = self.qvalues[state][action]
                if q > max_q:
                    max_q = q
                    best_action = action
            action = best_action

        return action

    def update(self, state, action, reward, next_state):

        future_q_value = 0
        if next_state in self.actions_par_etat:
            future_q_value = max(
                [self.qvalues[next_state][a] for a in self.actions_par_etat[next_state]]
            )

        target = reward + self.discount_factor * future_q_value

        q_s_a_difference = target - self.qvalues[state][action]

        self.qvalues[state][action] += self.lr * q_s_a_difference

        return q_s_a_difference

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
