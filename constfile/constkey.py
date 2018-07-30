# log config
LOG_ISFILEOUT = "log.fileout"                           # true: log out to file else to stdout
LOG_FILEPATH  = "log.filepath"                           # file path to log into
LOG_LEVEL     = "log.level"                             # level of log [CRITICAL,FATAL',ERROR',WARN',WARNING',INFO',DEBUG',NOTSET]



# model config

MODEL_SEED = "model.seed"
MODEL_STATE_COUNT = "model.state_count"
MODEL_ACTION_COUNT = "model.action_count"

MODEL_ACTOR_HIDDEN1 = "model.actor_hidden_size1"
MODEL_ACTOR_HIDDEN2 = "model.actor_hidden_size2"
MODEL_ACTOR_LR = "model.actor_learn_rate"
MODEL_ACTOR_WEIGHT_DECAY = "model.actor_weight_decay"

MODEL_CRITIC_HIDDEN1 = "model.critic_hidden_size1"
MODEL_CRITIC_HIDDEN2 = "model.critic_hidden_size2"
MODEL_CRITIC_LR = "model.critic_learn_rate"
MODEL_CRITIC_WEIGHT_DECAY = "model.critic_weight_decay"

MODEL_INIT_WEIGHT = "model.param_init_weight"

MODEL_BATCH_SIZE = "model.batch_size"
MODEL_TARGET_TAU = "model.target_tau"
MODEL_DISCOUNT = "model.discount"
MODEL_EPSILON = "model.epsilon"
MODEL_SAVE_PATH = "model.model_save"
MODEL_WARMUP = "model.warmup"
MODEL_SAVE_FREQ = "model.model_save_freq"

REPLAY_BUFFER_SIZE = "model.memory.limit"

RANDOM_MU = "model.random.mu"
RANDOM_THETA = "model.random.theta"
RANDOM_SIGMA = "model.random.sigma"

EVALUATOR_NUM_EPISODES = "evaluator.num_episodes"
EVALUATOR_VISABLE = "evaluator.env_visable"
EVALUATOR_DRAW = "evaluator.draw"
EVALUATOR_DRAW_PATH = "evaluator.draw_path"
EVALUATOR_MAX_STEP = "evaluator.max_steps"
