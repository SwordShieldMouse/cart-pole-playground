from algs import *

env = gym.make('CartPole-v0').unwrapped


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = Policy(4, 2).to(device)

optimizer = optim.Adam(policy.parameters(), lr = 1e-2)

episodes = 1000
gamma = 0.99

for episode in range(episodes):
    obs = env.reset()
    rewards = []
    times = []
    log_probs = Variable(torch.FloatTensor([]).to(device), requires_grad = True)
    for t in range(1000):
        env.render()
        obs = torch.FloatTensor(obs).to(device)

        # take an action
        logits = policy(obs)
        c = torch.distributions.Categorical(logits = logits)
        action = c.sample()
        log_probs = torch.cat([log_probs, torch.unsqueeze(c.log_prob(action), 0)])

        # get the next state and reward
        obs, reward, done, info = env.step(action.item())

        rewards.append(reward)

        if done:
            #print("Episode done after {} timesteps".format(t))
            times.append(t)
            break
    # do update after the episode
    T = len(rewards)

    for ix in range(T):
        rewards[ix] *= gamma ** ix

    returns = []
    #gammas = [gamma ** i for i in range(T)]

    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    # let autograd see the variables
    #rewards = Variable(torch.FloatTensor(rewards).to(device), requires_grad = True)
    #gammas = Variable(torch.FloatTensor(gammas).to(device), requires_grad = True)
    returns = torch.FloatTensor(returns).to(device)

    # the loss function is just the negative of the value function
    #loss = -torch.sum(torch.mul(torch.mul(rewards, gammas), log_probs))
    loss = -torch.sum(torch.mul(returns, log_probs))

    optimizer.zero_grad()
    loss.backward()
    #torch.autograd.gradcheck()
    #print(loss.grad)
    optimizer.step()

    if episode % 100 == 0 and episode > 0:
        print("On episode {}".format(episode + 1))
        print("Avg length of episode for last 100 episodes is {}".format(np.mean(times)))
        times = []
