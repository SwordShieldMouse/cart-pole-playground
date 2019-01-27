from algs import *

env = gym.make('CartPole-v0').unwrapped


policy = Policy(4, 2).to(device)
value_fn = Value_Function(4).to(device)

policy_optimizer = optim.Adam(policy.parameters(), lr = 1e-2)
value_fn_optimizer = optim.Adam(value_fn.parameters(), lr = 1e-2)

episodes = 1000
gamma = 0.99

#reinforce_baseline(env, policy, value_fn, policy_optimizer, value_fn_optimizer, episodes = episodes, T = 1000, gamma = gamma)
actor_critic(env, policy, value_fn, policy_optimizer, value_fn_optimizer, episodes = episodes, T = 1000, gamma = gamma)
