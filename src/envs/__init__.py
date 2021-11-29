from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from smac_full_action_space import StarCraft2FullActionSpaceEnv

import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2_full_action"] = partial(env_fn, env=StarCraft2FullActionSpaceEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
