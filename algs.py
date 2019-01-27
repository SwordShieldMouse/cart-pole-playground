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
            nn.Linear(env_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(p = 0.6),
            nn.Linear(128, action_dim),
            nn.LogSoftmax(dim = -1)
        )

    def forward(self, x):
        return self.layers(x)

# for estimate of value function v_\pi(s)
class Value_Function(nn.Module):
    def __init__(self, env_dim):
        super(Value_Function, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)


def reinforce_baseline(env, policy, value_fn, policy_optimizer, value_fn_optimizer, episodes = 1000, T = 1000, gamma = 0.99):
    for episode in range(episodes):
        obs = env.reset()
        rewards = []
        times = []
        state_history = [obs]
        log_probs = Variable(torch.FloatTensor([]).to(device), requires_grad = True)
        for t in range(T):
            env.render()
            obs = torch.FloatTensor(obs).to(device)

            # take an action
            # TODO: implement actor-critic
            logits = policy(obs)
            c = torch.distributions.Categorical(logits = logits)
            action = c.sample()
            log_probs = torch.cat([log_probs, torch.unsqueeze(c.log_prob(action), 0)])

            # get the next state and reward
            obs, reward, done, info = env.step(action.item())

            # history
            state_history.append(obs)
            rewards.append(reward)

            if done:
                times.append(t)
                break

        # do update after the episode

        # use a baseline
        baselines = Variable(torch.FloatTensor([value_fn(torch.FloatTensor(s).to(device)) for s in state_history]).to(device), requires_grad = True)

        for ix in range(len(rewards)):
            rewards[ix] *= gamma ** ix

        returns = []
        #gammas = [gamma ** i for i in range(T)]

        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        #print(log_probs.flip(0))
        log_probs = torch.flip(torch.cumsum(log_probs.flip(0), 0), [0])

        # let autograd see the variables
        #rewards = Variable(torch.FloatTensor(rewards).to(device), requires_grad = True)
        #gammas = Variable(torch.FloatTensor(gammas).to(device), requires_grad = True)
        returns = torch.FloatTensor(returns).to(device)

        # the policy loss function is just the negative of the sample of the value function
        # the value function loss is just mse
        policy_loss = -torch.sum(torch.mul(returns - baselines.narrow(0, 1, baselines.shape[0] - 1), log_probs))
        value_fn_loss = torch.sum((returns - baselines.narrow(0, 1, baselines.shape[0] - 1)) ** 2)

        policy_optimizer.zero_grad()
        value_fn_optimizer.zero_grad()

        policy_loss.backward()
        value_fn_loss.backward()

        policy_optimizer.step()
        value_fn_optimizer.step()

        if episode % 100 == 0 and episode > 0:
            print("On episode {}".format(episode + 1))
            print("Avg length of episode for last 100 episodes is {}".format(np.mean(times)))
            times = []


def actor_critic(env, policy, value_fn, policy_optimizer, value_fn_optimizer, episodes = 1000, T = 1000, gamma = 0.99):
    for episode in range(episodes):
        obs = env.reset()
        rewards = []
        times = []
        state_history = [obs]
        log_probs = Variable(torch.FloatTensor([]).to(device), requires_grad = True)
        for t in range(T):
            env.render()
            obs = torch.FloatTensor(obs).to(device)

            # take an action
            # TODO: implement actor-critic
            logits = policy(obs)
            c = torch.distributions.Categorical(logits = logits)
            action = c.sample()
            log_probs = torch.cat([log_probs, torch.unsqueeze(c.log_prob(action), 0)])

            # get the next state and reward
            obs, reward, done, info = env.step(action.item())

            # history
            state_history.append(obs)
            rewards.append(reward)

            # do the updates we can do during the episode
            delta = Variable(rewards[-1] + gamma * value_fn(torch.FloatTensor(state_history[-1])) - value_fn(torch.FloatTensor(state_history[-2])), requires_grad = True)

            policy_loss = -delta * log_probs[-1]
            policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph = True)
            policy_optimizer.step()

            value_fn_loss = delta ** 2
            value_fn_optimizer.zero_grad()
            value_fn_loss.backward()
            value_fn_optimizer.step()

            if done:
                times.append(t)
                break

        # do update after the episode

        # use a baseline
        baselines = Variable(torch.FloatTensor([value_fn(torch.FloatTensor(s).to(device)) for s in state_history]).to(device), requires_grad = True)

        for ix in range(len(rewards)):
            rewards[ix] *= gamma ** ix

        returns = []
        #gammas = [gamma ** i for i in range(T)]

        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        #print(log_probs.flip(0))
        log_probs = torch.flip(torch.cumsum(log_probs.flip(0), 0), [0])

        # let autograd see the variables
        #rewards = Variable(torch.FloatTensor(rewards).to(device), requires_grad = True)
        #gammas = Variable(torch.FloatTensor(gammas).to(device), requires_grad = True)
        returns = torch.FloatTensor(returns).to(device)

        # the policy loss function is just the negative of the sample of the value function
        policy_loss = -torch.sum(torch.mul(returns - baselines.narrow(0, 1, baselines.shape[0] - 1), log_probs))
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        if episode % 100 == 0 and episode > 0:
            print("On episode {}".format(episode + 1))
            print("Avg length of episode for last 100 episodes is {}".format(np.mean(times)))
            times = []
