from __future__ import absolute_import
from collections import  namedtuple
from utils.logsupport import log
from utils.configsupport import config
from constfile.constkey import *
import random

import numpy as np

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


class Memory(object):
    def __init__(self):

        limit = config.get(REPLAY_BUFFER_SIZE)

        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.current_observations = RingBuffer(limit)
        self.next_observations = RingBuffer(limit)

    def __len__(self):
        return len(self.actions)

    def sample_batch_indexes(self, low, high, size):
        if high - low >= size:
            r = range(low, high)
            batch_idxs = random.sample(r, size)
        else:
            log.warn('Not enough data in replaybuffer!!!')
            batch_idxs = np.random.random_integers(low, high - 1, size=size)
        assert len(batch_idxs) == size
        return batch_idxs

    def sample(self, batch_size):
        nb_entries = len(self.actions)
        batch_idxs = self.sample_batch_indexes(0, nb_entries - 1, size=batch_size)

        batch_idxs = np.array(batch_idxs)

        assert np.min(batch_idxs) >= 0
        assert np.max(batch_idxs) < nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal1 = self.terminals[idx]
            action = self.actions[idx]
            reward = self.rewards[idx]
            state0 = self.current_observations[idx]
            state1 = self.next_observations[idx]

            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size
        return experiences

    def sample_and_split(self, batch_size):
        experiences = self.sample(batch_size)

        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        state0_batch = np.array(state0_batch).reshape(batch_size, -1)
        state1_batch = np.array(state1_batch).reshape(batch_size, -1)
        terminal1_batch = np.array(terminal1_batch).reshape(batch_size, -1)
        reward_batch = np.array(reward_batch).reshape(batch_size, -1)
        action_batch = np.array(action_batch).reshape(batch_size, -1)

        return state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch

    def append(self, current_observation, action, next_observation, reward, terminal):
        self.current_observations.append(current_observation)
        self.actions.append(action)
        self.next_observations.append(next_observation)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    def append_experiences(self, experiences):
        self.current_observations.append(experiences.state0)
        self.actions.append(experiences.action)
        self.next_observations.append(experiences.state1)
        self.rewards.append(experiences.reward)
        self.terminals.append(experiences.terminal1)

