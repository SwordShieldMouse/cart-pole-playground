from algs import *

env = gym.make('CartPole-v0').unwrapped


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = Policy(4, 2).to(device)

optimizer = optim.Adam(policy.parameters(), lr = 1e-2)

episodes = 1000
gamma = 0.9

for episode in range(episodes):
    obs = env.reset()
    rewards = []
    times = []
    for t in range(1000):
        env.render()
        obs = torch.FloatTensor(obs).to(device)

        # take an action
        action = policy(obs).argmax()

        # get the next state and reward
        obs, reward, done, info = env.step(action.item())

        rewards.append(reward)

        if done:
            #print("Episode done after {} timesteps".format(t))
            times.append(t)
            break
    # do update after the episode
    T = len(rewards)
    rewards = Variable(torch.FloatTensor(rewards).to(device), requires_grad = True)
    gammas = [1]
    for i in range(1, T):
        gammas.append(gamma * rewards[i - 1])

    gammas = Variable(torch.FloatTensor(gammas).to(device), requires_grad = True)

    # the loss function is just the negative of the value function
    loss = -torch.sum(torch.mul(rewards, gammas))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 100 == 0:
        print("On episode {}".format(episode + 1))
        print("Avg length of episode for last 100 episodes is {}".format(np.mean(times)))
        times = []
