from typing import Dict
from .gail import GAILPolicy
from .ppo import PPOPolicy

def make_policy(policy_name: str, policy_config: Dict):
    if policy_name == 'gail':
        return GAILPolicy(policy_config)
    elif policy_name == 'ppo':
        return PPOPolicy(policy_config)
    else:
        raise ValueError(f"No supported algorighm {policy_name}")
