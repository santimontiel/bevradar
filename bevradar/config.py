from omegaconf import OmegaConf


def load_config(path: str) -> OmegaConf:
    """Load an experiment configuration file.

    Args:
        path (str): Path to the configuration file.

    Returns:
        OmegaConf: The configuration object.
    """
    OmegaConf.register_new_resolver("div", lambda x, y: int(x // y))
    OmegaConf.register_new_resolver("mul", lambda x, y: int(x * y))
    config = OmegaConf.load(path)

    if "base_configs" in config.keys():
        base_configs = [OmegaConf.load(c) for c in config.base_configs]
        config = OmegaConf.merge(*base_configs, config)

    return config