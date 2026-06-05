from .go2x5_config import Go2X5RoughCfg, Go2X5RoughCfgPPO


class Go2X5FtLiftCfg(Go2X5RoughCfg):
    """Backward-compatible alias for the B1Z1-aligned Go2X5 low-level config."""

    pass


class Go2X5FtLiftCfgPPO(Go2X5RoughCfgPPO):
    class runner(Go2X5RoughCfgPPO.runner):
        experiment_name = 'go2x5_ftlift'
