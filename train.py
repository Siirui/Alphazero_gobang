import random
import numpy as np
from collections import deque, defaultdict
from game import GameState, Game
from agents import MCTSPlayer
from model import PolicyValueNet
from MCTS_pure import MCTSPlayer as MCTS_Pure


class TrainPipeline(object):
    def __init__(self, init_model=None):
        self.grid_shape = (6, 6)
        self.learning_rate = 2e-3
        self.n = 4
        self.lr_multiplier = 1.0
        self.n_simulation = 400
        self.c_puct = 5
        self.temp = 1.0
        self.play_batch_size = 1
        self.buffer_size = 10000
        self.batch_size = 512
        self.epochs = 5
        self.kl_targ = 50
        self.check_freq = 50
        self.game_batch_num = 1
        self.best_win_ratio = 0.0
        self.pure_mcts_simulation_num = 1000
        self.episode_len = None
        self.game = Game(self.grid_shape, self.n)
        self.data_buffer = deque(maxlen=self.buffer_size)
        if init_model:
            self.policy_value_net = PolicyValueNet(
                self.grid_shape, model_file=init_model
            )
        else:
            self.policy_value_net = PolicyValueNet(
                self.grid_shape
            )
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_function,
            c_puct=self.c_puct,
            n_simulation=self.n_simulation,
            is_selfplay=1
        )

    def augment_data(self, play_data):
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                equal_state = np.array([np.rot90(s, i) for s in state])
                equal_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.grid_shape[0], self.grid_shape[1])), i)
                extend_data.append((equal_state, np.flipud(equal_mcts_prob).flatten(), winner))

                equal_state = np.array([np.fliplr(s) for s in equal_state])
                equal_mcts_prob = np.fliplr(equal_mcts_prob)
                extend_data.append((equal_state, np.flipud(equal_mcts_prob).flatten(), winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, is_shown=1, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            play_data = self.augment_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        kl = None
        new_probs, new_v = None, None
        loss, entropy = None, None
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learning_rate * self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1))
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        print(f"kl: {kl:.5f}, lr_multiplier:{self.lr_multiplier:.3f}, "
              f"loss:{loss}, entropy:{entropy}, explained_var_old:{explained_var_old:.3f}, "
              f"explained_var_new:{explained_var_new:.3f}")
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_function,
                                         c_puct=self.c_puct,
                                         n_simulation=self.n_simulation)
        pure_mcts_player = MCTS_Pure(c_puct=5, n_simulation=self.pure_mcts_simulation_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player, pure_mcts_player,
                                          start_player=(1 if i % 2 == 0 else -1),
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5 * win_cnt[2]) / n_games
        print(f"num_playouts:{self.pure_mcts_simulation_num}, win:{win_cnt[1]}, lose:{win_cnt[-1]}, "
              f"tie:{win_cnt[2]}")
        return win_ratio

    def run(self):
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print(f"batch i:{i + 1}, episode_len:{self.episode_len}")
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                if (i + 1) % self.check_freq == 0:
                    print(f"current self-play batch: {i + 1}")
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model("./current_policy.model")
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!")
                        self.policy_value_net.save_model('./best_policy.mode')
                        if self.best_win_ratio == 1.0 and self.pure_mcts_simulation_num < 5000:
                            self.pure_mcts_simulation_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()


