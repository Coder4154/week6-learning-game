# play_game.py
from tic_tac_toe import TicTacToe
from q_learning_agent import QLearningAgent

def play():
    game = TicTacToe()
    ai = QLearningAgent(player_id=-1)
    ai.training = False
    ai.load()

    state = game.reset()

    while True:
        game.display()

        # Human move (X = 1)
        move = input("Enter your move as row,col (e.g. 0,2): ")
        r, c = map(int, move.split(","))
        game.make_move((r, c), 1)

        if game.check_winner() is not None:
            break

        # AI move (O = -1)
        state = game.get_state()
        action = ai.choose_action(state, game.get_available_actions())
        game.make_move(action, -1)

        if game.check_winner() is not None:
            break

    game.display()
    print("Game Over! Winner:", game.check_winner())

if __name__ == "__main__":
    play()
