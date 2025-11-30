# train_agent.py
from tic_tac_toe import TicTacToe
from q_learning_agent import QLearningAgent

print("TRAIN_AGENT STARTED")

EPISODES = 50000

def train():
    game = TicTacToe()
    agent_x = QLearningAgent(player_id=1)
    agent_o = QLearningAgent(player_id=-1)

    with open("results/training_progress.txt", "w") as log:
        for episode in range(EPISODES):
            state = game.reset()
            done = False
            current_agent = agent_x

            while not done:
                actions = game.get_available_actions()
                action = current_agent.choose_action(state, actions)
                game.make_move(action, current_agent.player_id)

                next_state = game.get_state()
                winner = game.check_winner()
                next_actions = game.get_available_actions()

                if winner is not None:
                    # Reward logic
                    if winner == 0:
                        reward = 0  # tie
                    else:
                        reward = 1 if winner == current_agent.player_id else -1

                    current_agent.update_q_value(state, action, reward, next_state, next_actions)
                    done = True
                    continue

                current_agent.update_q_value(state, action, 0, next_state, next_actions)

                # Switch player
                current_agent = agent_o if current_agent == agent_x else agent_x
                state = next_state

            if episode % 1000 == 0:
                log.write(f"Episode {episode} completed\n")
                print(f"Episode {episode}...")

    agent_x.save()
    agent_o.save()
    print("Training completed!")

if __name__ == "__main__":
    train()
