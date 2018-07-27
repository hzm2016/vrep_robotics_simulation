import json
import numpy as np

import gym
import gym_vrep # must not delete this import
import tensorflow as tf

from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
from ReplayBuffer import ReplayBuffer

OU = OU()  # Ornstein-Uhlenbeck Process

RUN = 2
SETTING = "vanilla"

TOP_EPISODE_BUFFER = 50

def playGame(train_indicator=1, render=True, debug=True):  # 1 means Train, 0 means simply Run
    if train_indicator == 1:
        log = open("rewards/rewards-{}-{}.log".format(SETTING, RUN), 'w')

    BUFFER_SIZE = 10000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.0001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic

    action_dim = 6  # 6 motors
    state_dim = 6  # of sensors input

    # np.random.seed(1337)

    EXPLORE = 10000.
    episode_count = 5000
    max_steps = 10
    step = 0
    epsilon = 1

    top_episodes = np.zeros((TOP_EPISODE_BUFFER, max_steps, action_dim), np.float32)
    top_episode_rewards = np.zeros((TOP_EPISODE_BUFFER), np.float32)

    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    env = gym.make('ErgoBall-v0')

    env.env._actualInit(headless=True)

    if train_indicator == 0:
        try:
            actor.model.load_weights("models/actormodel.h5")
            critic.model.load_weights("models/criticmodel.h5")
            actor.target_model.load_weights("models/actormodel.h5")
            critic.target_model.load_weights("models/criticmodel.h5")
            print("Weight load successfully")
        except:
            print("Cannot find the weight")

    print("Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        ob = env.reset()

        s_t = ob

        ep_buffer = np.zeros((max_steps, action_dim), np.float32)

        total_reward = 0.
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))

            for k in range(6):
                noise_t[0][k] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][k], 0.0, 2.0, 1.0)
                a_t[0][k] = a_t_original[0][k] + noise_t[0][k]

            if render:
                env.render()

            ep_buffer[j] = a_t[0]

            actions = tuple(np.array([action]) for action in a_t[0].tolist())

            ob, r_t, done, info = env.step(actions)

            s_t1 = ob

            buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer

            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            if (train_indicator):
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            if debug:
                print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

        lowest_top_reward = top_episode_rewards.min()
        if total_reward > lowest_top_reward:
            lowest_top_reward_idx = top_episode_rewards.argmin()
            top_episodes[lowest_top_reward_idx] = ep_buffer
            top_episode_rewards[lowest_top_reward_idx] = total_reward

        if np.mod(i, 3) == 0:
            if train_indicator == 1:
                print("Now we save model")
                actor.model.save_weights("models/actormodel.h5", overwrite=True)
                with open("models/actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("models/criticmodel.h5", overwrite=True)
                with open("models/criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

                np.savez("top-episodes/actions-{}-{}.npz".format(SETTING, RUN), top_episodes)
                np.savez("top-episodes/rewards-{}-{}.npz".format(SETTING, RUN), top_episode_rewards)

        print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")
        if train_indicator == 1:
            log.write('{}\n'.format(total_reward))
            log.flush()

    env.close()
    if train_indicator == 1:
        log.close()
    print("Finish.")


if __name__ == "__main__":
    playGame(train_indicator=1, render=False, debug=True)
