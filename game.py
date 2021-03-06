import numpy as np
import logging


class GameState(object):
    def __init__(self, board, player_turn, steps, grid_shape, n):
        self.display_board = board
        self.steps = steps
        self.n = n
        self.grid_shape = grid_shape
        self.player_turn = player_turn
        self.board = self._convertNumber2binary()
        self.id = self._convertState2Id()
        # self.binary = self._binary()
        self.allowedActions = list(range(self.grid_shape[0] * self.grid_shape[1]))
        # self.isEndGame = self._checkForEndGame()

    def get_current_state(self):
        """
        获取当前状态的局面描述 4*15*15的二值平面
        第1个平面为当前player的棋子位置
        第2个平面为对手player的棋子位置
        第3个平面为对手player最近一步的棋子位置
        第4个平面为棋子的颜色, 黑棋为1，白棋为0
        """
        state = np.zeros((4, self.grid_shape[0], self.grid_shape[1]))
        current_player_position = np.zeros(len(self.board), dtype=np.int32)
        current_player_position[self.board == self.player_turn] = 1
        other_player_position = np.zeros(len(self.board), dtype=np.int32)
        other_player_position[self.board == -self.player_turn] = 1
        state[0] = current_player_position.reshape(self.grid_shape[0], self.grid_shape[1])
        state[1] = other_player_position.reshape(self.grid_shape[0], self.grid_shape[1])
        if self.steps > 1:
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    if self.display_board[i * self.grid_shape[1] + j] == self.steps - 1:
                        state[2][i, j] = 1
        if self.player_turn == 1:
            state[3][:, :] = 1.0
        return state[:, ::-1, :]

    def _convertNumber2binary(self):
        """将当前的奇偶棋子转换为正负棋子方便计算，减小常数"""
        new_board = np.zeros(self.grid_shape[0] * self.grid_shape[1], dtype=np.int32)
        # new_board[self.display_board > 0 and self.display_board % 2 == 1] = 1
        # new_board[self.display_board > 0 and self.display_board % 2 == 0] = -1
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                value = self.display_board[i * self.grid_shape[1] + j]
                if value > 0:
                    if value % 2 == 1:
                        new_board[i * self.grid_shape[1] + j] = 1
                    else:
                        new_board[i * self.grid_shape[1] + j] = -1
        return new_board

    def _binary(self):
        """
        将棋盘二值化，经过Number2binary转换后的棋盘中1为黑棋，-1为白棋
        player_turn=1表示黑方下棋，player_turn=-1表示白方下棋
        Returns:
        """
        current_player_position = np.zeros(len(self.board), dtype=np.int32)
        # for index in range(len(self.board)):
        #     if self.board[index] > 0 and (self.board[index] % 2) ^ self.player_turn == 0:
        #         current_player_position[index] = 1
        current_player_position[self.board == self.player_turn] = 1

        other_player_position = np.zeros(len(self.board), dtype=np.int32)
        # for index in range(len(self.board)):
        #     if self.board[index] > 0 and (self.board[index] % 2) ^ self.player_turn == 1:
        #         other_player_position[index] = 1
        other_player_position[self.board == -self.player_turn] = 1

        position = np.append(current_player_position, other_player_position)
        return position

    def _convertState2Id(self):
        current_player_position = np.zeros(len(self.board), dtype=np.int32)
        current_player_position[self.board == 1] = 1

        other_player_position = np.zeros(len(self.board), dtype=np.int32)
        other_player_position[self.board == -1] = 1

        position = np.append(current_player_position, other_player_position)
        _id = ''.join(map(str, position))
        return _id

    def checkForEndGame(self):
        """判断当前棋局是否结束并返回赢家是谁"""
        pos_x, pos_y = None, None
        for i in range(self.grid_shape[0]):
            if pos_x is not None and pos_y is not None:
                break
            for j in range(self.grid_shape[1]):
                if self.display_board[i * self.grid_shape[1] + j] == self.steps - 1:
                    pos_x, pos_y = i, j
                    break

        color = self.board[pos_x * self.grid_shape[1] + pos_y]
        if color == 0:
            return 0
        # 检查横向
        count = 1
        for i in range(1, self.n):
            if pos_y + i >= self.grid_shape[1] or color != self.board[pos_x * self.grid_shape[1] + pos_y + i]:
                break
            else:
                count += 1
        for i in range(1, self.n):
            if pos_y - i < 0 or color != self.board[pos_x * self.grid_shape[1] + pos_y - i]:
                break
            else:
                count += 1
        if count == self.n:
            return -self.player_turn

        # 检查纵向
        count = 1
        for i in range(1, self.n):
            if pos_x + i >= self.grid_shape[0] or color != self.board[(pos_x + i) * self.grid_shape[1] + pos_y]:
                break
            else:
                count += 1
        for i in range(1, self.n):
            if pos_x - i < 0 or color != self.board[(pos_x - i) * self.grid_shape[1] + pos_y]:
                break
            else:
                count += 1
        if count == self.n:
            return -self.player_turn

        # 检查左上到右下对角线
        count = 1
        for i in range(1, self.n):
            if pos_x + i >= self.grid_shape[0] or pos_y + i >= self.grid_shape[1] or color != self.board[(pos_x + i) * self.grid_shape[1] + pos_y + i]:
                break
            else:
                count += 1
        for i in range(1, self.n):
            if pos_x - i < 0 or pos_y - i < 0 or color != self.board[(pos_x - i) * self.grid_shape[1] + pos_y - i]:
                break
            else:
                count += 1
        if count == self.n:
            return -self.player_turn

        # 检查右上到左下对角线
        count = 1
        for i in range(1, self.n):
            if pos_x + i >= self.grid_shape[0] or pos_y - i < 0 or color != self.board[(pos_x + i) * self.grid_shape[1] + pos_y - i]:
                break
            else:
                count += 1
        for i in range(1, self.n):
            if pos_x - i < 0 or pos_y + i >= self.grid_shape[1] or color != self.board[(pos_x - i) * self.grid_shape[1] + pos_y + i]:
                break
            else:
                count += 1
        if count == self.n:
            return -self.player_turn
        # # 检查横向
        # for i in range(self.grid_shape[0]):
        #     for j in range(0, self.grid_shape[1] - self.n + 1):
        #         count = 0
        #         for k in range(self.n):
        #             count += self.board[i * self.grid_shape[1] + j + k]
        #         if abs(count) == self.n:
        #             return count // self.n
        # # 检查纵向
        # for i in range(0, self.grid_shape[0] - self.n + 1):
        #     for j in range(self.grid_shape[0]):
        #         count = 0
        #         for k in range(self.n):
        #             count += self.board[(i + k) * self.grid_shape[1] + j]
        #         if abs(count) == self.n:
        #             return count // self.n
        # # 检查左上到右下的对角线
        # for i in range(0, self.grid_shape[0] - self.n + 1):
        #     for j in range(0, self.grid_shape[1] - self.n + 1):
        #         count = 0
        #         for k in range(self.n):
        #             count += self.board[(i + k) * self.grid_shape[1] + j + k]
        #         if abs(count) == self.n:
        #             return count // self.n
        #
        # # 检查右上到左下的对角线
        # for i in range(0, self.grid_shape[0] - self.n + 1):
        #     for j in range(self.n - 1, self.grid_shape[1]):
        #         count = 0
        #         for k in range(self.n):
        #             count += self.board[(i + k) * self.grid_shape[1] + j - k]
        #         if abs(count) == self.n:
        #             return count // self.n

        if len(self.allowedActions) == 0:
            return 2  # tie
        # 未结束
        return 0

    def takeAction(self, action):
        #new_display_board = np.array(self.display_board)
        self.display_board[action] = self.steps
        self.allowedActions.remove(action)
        # new_state = GameState(new_display_board, -self.player_turn, self.steps + 1, self.grid_shape, self.n)
        self.board = self._convertNumber2binary()
        self.steps += 1
        self.player_turn = -self.player_turn
        #return new_state


class Game(object):
    def __init__(self, grid_shape=(8, 8), n=5):
        self.grid_shape = grid_shape
        self.n = n
        self.gameState = GameState(np.zeros(self.grid_shape[0] * self.grid_shape[1], dtype=np.int32), 1, 1, self.grid_shape, n=n)
        self.actionSpace = np.zeros(self.grid_shape[0] * self.grid_shape[1], dtype=np.int32)
        self.name = "Gobang"

    def reset(self, start_player=1):
        self.gameState = GameState(np.zeros(self.grid_shape[0] * self.grid_shape[1], dtype=np.int32), 1, 1, self.grid_shape, self.n)
        self.currentPlayer = start_player
        return self.gameState
    
    def graphic(self, player1, player2):
        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(self.grid_shape[1]):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(self.grid_shape[0]):
            print("{0:4d}".format(i), end='')
            for j in range(self.grid_shape[1]):
                loc = i * self.grid_shape[1] + j
                if self.gameState.display_board[loc] == 0:
                    print('_'.center(8), end='')
                else:
                    print(('X' if self.gameState.display_board[loc] % 2 == 1 else 'O').center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=1, is_shown=1):
        if start_player not in (1, -1):
            raise Exception('start_player should be either 1 (player1 first) or -1 (player2 first)')
        self.reset(start_player)
        player1.set_player_ind(1)
        player2.set_player_ind(-1)
        players = {1: player1, -1: player2}
        if is_shown:
            self.graphic(player1.player, player2.player)
        while True:
            player_in_turn = players[self.gameState.player_turn]
            move = player_in_turn.get_action(self.gameState)
            self.gameState.takeAction(move)
            if is_shown:
                self.graphic(player1.player, player2.player)
            winner = self.gameState.checkForEndGame()
            if winner != 0:
                if is_shown:
                    if winner == 2:
                        print("Game end. Tie!")
                    else:
                        print("Game end. Winner is", players[winner])
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        self.reset(start_player=1)
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.gameState, temp=temp, return_prob=1)

            states.append(self.gameState.get_current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.currentPlayer)
            self.gameState.takeAction(move)

            if is_shown:
                self.graphic(1, -1)

            winner = self.gameState.checkForEndGame()
            if winner != 0:
                winners_z = np.zeros(len(current_players))
                if winner != 2:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player.reset_player()
                if is_shown:
                    if winner == 2:
                        print("Game end. Tie!")
                    else:
                        print(f"Game end. Winner is: {winner}")
                return winner, zip(states, mcts_probs, winners_z)







# agent = Game()
# agent.gameState, _, _ = agent.gameState.takeAction(0)
# agent.graphic("test1", "test2")
# state_ = agent.gameState.get_current_state()
