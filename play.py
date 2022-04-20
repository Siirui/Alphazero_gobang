import pickle
from game import GameState, Game
from agents import MCTSPlayer, HumanPlayer
from model import PolicyValueNet


def run():
    n = 5
    grid_shape = (15, 15)
    model_file = "best_policy.mode"
    try:
        game = Game(grid_shape, n)
        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')
        best_policy = PolicyValueNet(grid_shape, policy_param)
        mcts_player = MCTSPlayer(best_policy.policy_value_function,
                                 c_puct=5,
                                 n_simulation=400)
        human = HumanPlayer()
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == "__main__":
    run()
