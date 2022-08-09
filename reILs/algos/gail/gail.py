from reILs.envs.env_maker import make_env
from reILs.infrastructure.execution import WorkerSet

class GAILAgent:

    def __init__(self, config: Dict):

        self.obs_dim = config['obs_dim']
        self.act_dim = config['act_dim']
        self.max_act = config['max_act']

        print("Observations shape:", args.state_shape)
        print("Actions shape:", args.action_shape)
        print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

        self.worker_set = WorkerSet
