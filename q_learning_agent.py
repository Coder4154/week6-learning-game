# q_learning_agent.py
import random
import json
import os

class QLearningAgent:

    def __init__(self, player_id, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.player_id = player_id
        self.q_table = {}  # Stores Q-values
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.training = True

    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair"""
        key = f"{state}_{action}"
        if key not in self.q_table:
            self.q_table[key] = 0.0
        return self.q_table[key]

    def choose_action(self, state, available_actions):
        """Epsilon-greedy choice"""
        if not available_actions:
            return None

        # Explore
        if self.training and random.random() < self.exploration_rate:
            return random.choice(available_actions)

        # Exploit
        best_value = float("-inf")
        best_action = random.choice(available_actions)

        for action in available_actions:
            value = self.get_q_value(state, action)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def update_q_value(self, state, action, reward, next_state, next_actions):
        """Apply the Q-Learning update rule"""
        key = f"{state}_{action}"

        old_q = self.get_q_value(state, action)

        if next_actions:
            next_q = max(self.get_q_value(next_state, a) for a in next_actions)
        else:
            next_q = 0

        new_q = old_q + self.learning_rate * (
            reward + self.discount_factor * next_q - old_q
        )

        self.q_table[key] = new_q

    def save(self, filename="results/q_table.json"):
        with open(filename, "w") as f:
            json.dump(self.q_table, f)

    def load(self, filename="results/q_table.json"):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                self.q_table = json.load(f)
