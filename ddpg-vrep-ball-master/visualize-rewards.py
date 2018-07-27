import matplotlib.pyplot as plt
import numpy as np

RUN = 2
SETTING = "vanilla"

FIRST_N = -1

suffix = "all"
rewards = np.loadtxt("rewards/rewards-{}-{}.log".format(SETTING, RUN), dtype=np.float32)
if FIRST_N != -1:
    rewards = rewards[:FIRST_N]
    suffix = FIRST_N
x = np.arange(len(rewards))

plt.plot(x, rewards)
# plt.show()
plt.savefig('figures/figure-{}-{}-{}.png'.format(SETTING, RUN, suffix))
