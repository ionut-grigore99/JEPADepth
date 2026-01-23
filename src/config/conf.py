from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Conf:
    name: str = 'config.yaml'

    def __post_init__(self):
        with open(Path(__file__).parent / self.name, 'r') as f:
            self.conf = yaml.safe_load(f)
