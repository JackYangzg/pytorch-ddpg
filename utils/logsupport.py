import logging
from utils.configsupport import config
from constfile.constkey import *


__fileout = config.get(LOG_ISFILEOUT)
__filepath = config.get(LOG_FILEPATH)
__level = config.get(LOG_LEVEL)

__formated = "%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s"
__datefmt= '%a, %d %b %Y %H:%M:%S'

if not __fileout:
    logging.basicConfig(level=__level, format=__formated, datefmt=__datefmt)
else:
    logging.basicConfig(level=__level, format=__formated, datefmt=__datefmt, filename=__filepath)

log = logging.getLogger("ddpg")