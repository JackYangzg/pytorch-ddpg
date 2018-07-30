from utils import libconf
import os, sys

class ConfigSupport(object):
    def __init__(self):
        self.__config = self.__load()

    def __load(self):
        config = None
        path = os.path.join(sys.path[0], "..", "resources", "ddpg.conf")
        if not os.path.exists(path):
            path = os.path.join(sys.path[0], "resources", "ddpg.conf")

        with open(path, "r", encoding='utf-8') as f:
            config = libconf.load(f)

        return config

    def get(self, key):
        """
        :param key:
        :return:
        """
        if isinstance(key, str):
            key_split = key.split(".")
        else:
            key_split = [key]

        value = self.__config

        for sub_key in key_split:
            value = value[sub_key]

        return value

config = ConfigSupport()


if __name__ == "__main__":
    print(config.get("log.fileout"))

