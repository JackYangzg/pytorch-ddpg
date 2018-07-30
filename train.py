import time

from memory import Memory
from constfile.constkey import *
from constfile.ddpgqueue import fifo, model_queue
from ddpg import DDPG
from utils.configsupport import config
from utils.logsupport import log


class train(object):
    def __init__(self):
        self.model = DDPG()
        self.model.load_weights()
        self.memory = Memory()
        self.queue = fifo
        self.model_queue = model_queue
        self.warmup = config.get(MODEL_WARMUP)
        self.modelsavefreq = config.get(MODEL_SAVE_FREQ) * 1000


    def queue_pop(self):
        if not self.queue.empty():
            data = self.queue.get()
            self.memory.append_experiences(data)

    def train_model(self):
        while True:
            start = time.time()
            if len(self.memory) < self.warmup:
                log.warn("not enough samples[{}] in replay buffer!".format(len(self.memory)))
                time.sleep(10)
                start = time.time()
            else:
                end = time.time()
                if end-start > self.modelsavefreq:
                    self.model.save_model()
                    start = end

                self.model.update_policy(self.memory)
                actor, critic = self.model.get_model()

                if self.model_queue.empty():
                    self.model_queue.put((actor, critic))

    def draw_loss(self):
        actor_loss,critic_loss = self.model.get_loss()
        pass
