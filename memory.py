import numpy as np
from collections import deque

# import config


class Memory(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.long_term_memory = deque(maxlen=memory_size)
        self.short_term_memory = deque(maxlen=memory_size)

    def commit_short_term_memory(self, identities, state, action_values):
        for r in identities(state, action_values):
            self.short_term_memory.append({
                "board": r[0].board,
                "state": r[0],
                "id": r[0].id,
                "ActionValues": r[1],
                "playerTurn": r[0].playerTurn
            })

    def clear_short_term_memory(self):
        self.short_term_memory = deque(maxlen=self.memory_size)

    def commit_long_term_memory(self,):
        for i in self.short_term_memory:
            self.long_term_memory.append(i)
        self.clear_short_term_memory()
