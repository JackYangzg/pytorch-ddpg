import os

import numpy as np
import matplotlib.pyplot as plt
from ddpg import DDPG
from utils.configsupport import config
from constfile.constkey import *
from constfile.ddpgqueue import fifo, model_queue
from memory import Experience
from util import *

class Evaluator(object):

    def __init__(self):
        self.model = DDPG()
        self.num_episodes = config.get(EVALUATOR_NUM_EPISODES)
        self.max_step = config.get(EVALUATOR_MAX_STEP)
        self.visualize = config.get(EVALUATOR_VISABLE)
        self.save = config.get(EVALUATOR_DRAW)
        self.save_path = config.get(EVALUATOR_DRAW_PATH)
        self.queue = fifo
        self.model_queue = model_queue
        self.model.load_weights()

    def policy(self, observation):
        if not self.model_queue.empty():
            actor_state, critic_state = self.model_queue.get()
            self.model.load_state_dict(actor_state, critic_state)
        return 0

    def __call__(self, env):
        result = []
        for episode in range(self.num_episodes):

            # reset at the start of episode
            current_obs = env.reset()
            next_obs = None
            episode_steps = 0
            episode_reward = 0.
                
            assert current_obs is not None

            # start episode
            done = False
            while not done:
                if next_obs is not None:
                    current_obs = next_obs
                # basic operation, action ,reward, blablabla ...
                action = self.policy(current_obs)
                next_obs, reward, done, info = env.step(action)

                if next_obs is None:
                    current_expr = Experience(current_obs, action, reward, next_obs, True)
                else:
                    current_expr = Experience(current_obs, action, reward, next_obs, False)

                # put data into the queue
                self.queue.put(current_expr)
                # only max steps
                if self.max_step == "None" or episode_steps >= self.max_step:
                    done = True

                if self.visualize:
                    env.render(mode='human')

                # update
                episode_reward += reward
                episode_steps += 1

            result.append(episode_reward)

        result = np.array(result).reshape(-1, 1)

        if self.save:
            self.save_results(result, os.path.join('{}'.format(self.save_path), "validate_reward"))
        return np.mean(result)

    def save_results(self, result, fn):

        y = np.mean(result, axis=0)
        error = np.std(result, axis=0)
                    
        x = range(0, result.shape[1])
        fig, ax = plt.subplots(1, 1, figsize=(20, 18))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'.png')