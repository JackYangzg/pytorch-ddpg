import os

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from model import (Actor, Critic)
from memory import Memory
from random_process import OrnsteinUhlenbeckProcess
from util import *
from constfile.constkey import *
from utils.configsupport import config


class DDPG(object):
    def __init__(self):

        # random seed for torch
        __seed = config.get(MODEL_SEED)
        self.policy_loss = []
        self.critic_loss = []
        if __seed > 0:
            self.seed(__seed)

        self.nb_states  = config.get(MODEL_STATE_COUNT)
        self.nb_actions = config.get(MODEL_ACTION_COUNT)
        
        # Create Actor and Critic Network
        actor_net_cfg = {
            'hidden1': config.get(MODEL_ACTOR_HIDDEN1),
            'hidden2': config.get(MODEL_ACTOR_HIDDEN2),
            'init_w': config.get(MODEL_INIT_WEIGHT)
        }
        critic_net_cfg = {
            'hidden1': config.get(MODEL_CRITIC_HIDDEN1),
            'hidden2': config.get(MODEL_CRITIC_HIDDEN2),
            'init_w': config.get(MODEL_INIT_WEIGHT)
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **actor_net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **actor_net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=config.get(MODEL_ACTOR_LR),
                                weight_decay=config.get(MODEL_ACTOR_WEIGHT_DECAY))

        self.critic = Critic(self.nb_states, self.nb_actions, **critic_net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **critic_net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=config.get(MODEL_CRITIC_LR),
                                weight_decay=config.get(MODEL_CRITIC_WEIGHT_DECAY))

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = Memory()

        self.random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions,
                                                       theta=config.get(RANDOM_THETA),
                                                       mu=config.get(RANDOM_MU),
                                                       sigma=config.get(RANDOM_SIGMA))

        # Hyper-parameters
        self.batch_size = config.get(MODEL_BATCH_SIZE)
        self.tau = config.get(MODEL_TARGET_TAU)
        self.discount = config.get(MODEL_DISCOUNT)
        self.depsilon = 1.0 / config.get(MODEL_EPSILON)

        self.model_path = config.get(MODEL_SAVE_PATH)

        # 
        self.epsilon = 1.0

        # init device
        self.device_init()

    def update_policy(self, memory):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([to_tensor(next_state_batch),
                                                self.actor_target(to_tensor(next_state_batch))])

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])
        value_loss = F.mse_loss(q_batch, target_q_batch)

        value_loss.backward()
        self.critic_optim.step()
        self.critic_loss.append(value_loss.data[0])

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()
        self.policy_loss.append(policy_loss.data[0])

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def get_loss(self):
        return self.policy_loss, self.critic_loss

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def device_init(self):
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)


    def random_action(self):
        action = np.random.uniform(-1., 1., self.nb_actions)
        return action

    def select_action(self, s_t):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)

        action += max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, -1., 1.)

        return action

    def clean(self, decay_epsilon):
        if decay_epsilon:
            self.epsilon -= self.depsilon

    def reset(self):
        self.random_process.reset_states()

    def load_weights(self):
        if not os.path.exists(self.model_path):
            return

        actor_path = os.path.exists(os.path.join(self.model_path, 'actor.pkl'))
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path))

        critic_path = os.path.exists(os.path.join(self.model_path, 'critic.pkl'))
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path))


    def save_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        actor_path = os.path.exists(os.path.join(self.model_path, 'actor.pkl'))
        torch.save(self.actor.state_dict(), actor_path)

        critic_path = os.path.exists(os.path.join(self.model_path, 'critic.pkl'))
        torch.save(self.critic.state_dict(), critic_path)

    def get_model(self):
        return self.actor.state_dict(), self.critic.state_dict()

    def load_state_dict(self, actor_state, critic_state):
        self.actor.load_state_dict(actor_state)
        self.critic.load_state_dict(critic_state)

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
