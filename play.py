import pickle
from game import GameState, Game
from agents import MCTSPlayer, HumanPlayer
from model import PolicyValueNet


def run():
    n = 5
    grid_shape = (9, 9)
    model_file = "best_policy_9.model"
    try:
        game = Game(grid_shape, n)
        best_policy = PolicyValueNet(grid_shape, model_file=model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_function,
                                 c_puct=5,
                                 n_simulation=400)
        human = HumanPlayer(grid_shape)
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == "__main__":
    run()
