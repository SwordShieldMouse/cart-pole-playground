from includes import *

# represent a single transition in env
# just the standard (state, action, next state, reward) tuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# holds recently observed transitions so that network can reuse them
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity # overwrite the memory

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
