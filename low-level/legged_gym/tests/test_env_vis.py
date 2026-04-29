import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


def test_env_vis(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1

    args.headless = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    for _ in range(int(3 * env.max_episode_length)):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        env.step(actions)

    print("Done")


if __name__ == '__main__':
    args = get_args()
    test_env_vis(args)
