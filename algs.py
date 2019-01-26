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

class DQN(nn.Module):
    def __init__(self, h, w):
        super(DQN, self).__init__()

        self.convs = nn.ModuleList([nn.utils.weight_norm(nn.Conv2d(3, 16, kernel_size=5, stride=2)),
                                    nn.utils.weight_norm(nn.Conv2d(16, 32, kernel_size=5, stride=2)),
                                    nn.utils.weight_norm(nn.Conv2d(32, 32, kernel_size=5, stride=2))
                                    ])
        #self.bn3 = nn.BatchNorm2d(32)

        conv_h = h
        conv_w = w
        for _ in range(len(convs)):
            conv_h = get_conv_dim(conv_h)
            conv_w = get_conv_dim(conv_w)

        linear_input_size = conv_h * conv_w * 32
        self.head = nn.Linear(linear_input_size, 2)

    def forward(self, x):
        for conv in self.convs:
            x = F.leaky_relu(conv(x))
        return self.head(x.view(x.size(0), -1))

    def get_conv_dim(size, kernel_size = 5, stride = 2):
        return (size - kernel_size) // stride + 1

# for policy gradient
class Policy(nn.Module):
    def __init__(self, env_dim, action_dim):
        super(Policy, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, action_dim),
            nn.LogSoftmax(dim = -1)
        )

    def forward(self, x):
        return self.layers(x)
