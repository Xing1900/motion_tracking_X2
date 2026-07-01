import os
from .humanoid import G1_CFG, G1_COL_FULL, G1_COL_FULL_SELF
from .X2.humanoid import X2_CFG, X2_ULTRA, X2_ULTRA_SELF

ASSET_PATH = os.path.dirname(__file__)

ROBOTS = {
    "g1": G1_CFG,
    "g1_col_full": G1_COL_FULL,
    "g1_col_full_self": G1_COL_FULL_SELF,
    "x2": X2_CFG,
    "x2_ultra": X2_ULTRA,
    "x2_ultra_self": X2_ULTRA_SELF,
}


def get_robot_cfg(name: str):
    if name not in ROBOTS:
        raise ValueError(f"Unknown robot name: {name}")
    return ROBOTS[name]
