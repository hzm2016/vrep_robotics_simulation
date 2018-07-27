import gym
import gym_vrep
import numpy as np
import time

RUN = 2
SETTING = "vanilla"

TOP_N = 10

rewards = np.load("rewards-{}-{}.npz".format(SETTING, RUN))['arr_0']
actions = np.load("actions-{}-{}.npz".format(SETTING, RUN))['arr_0']

top_n_idx = np.argpartition(rewards, -TOP_N)[-TOP_N:]

env = gym.make('ErgoBall-v0')
env.env._actualInit(headless=False)

for ep in range(TOP_N):
    env.reset()
    print "TOP {} EPISODE".format(ep+1)

    for act in range(len(actions[0,:,0])):
        action = actions[ep, act]
        action_conv = tuple(np.array([a]) for a in action.tolist())
        _, reward, _, _ = env.step(action_conv)
        print "reward: {}".format(reward)
        # time.sleep(.5)

