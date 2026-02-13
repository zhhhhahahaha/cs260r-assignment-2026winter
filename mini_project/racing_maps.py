"""Racing map variants for local evaluation and training.

Defines additional track geometries beyond the default "circuit" map.
Use set_racing_map() before creating an env to switch maps.
"""

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.straight import Straight
from metadrive.constants import PGLineType
from metadrive.envs.marl_envs.marl_racing_env import RacingMap


def _build_track(pg_map, specs):
    """Shared helper: given a PGMap instance and a list of block specs, build the track."""
    parent_node_path = pg_map.engine.worldNP
    physics_world = pg_map.engine.physics_world

    LANE_NUM = pg_map.config["lane_num"]
    LANE_WIDTH = pg_map.config["lane_width"]

    last_block = FirstPGBlock(
        pg_map.road_network, lane_width=LANE_WIDTH, lane_num=LANE_NUM,
        render_root_np=parent_node_path, physics_world=physics_world,
        remove_negative_lanes=True,
        side_lane_line_type=PGLineType.GUARDRAIL, center_line_type=PGLineType.GUARDRAIL,
    )
    pg_map.blocks.append(last_block)

    block_index = 1
    for kind, params in specs:
        cls = Straight if kind == "straight" else Curve
        last_block = cls(
            block_index, last_block.get_socket(0), pg_map.road_network, 1,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.GUARDRAIL, center_line_type=PGLineType.GUARDRAIL,
        )
        last_block.construct_from_config(params, parent_node_path, physics_world)
        pg_map.blocks.append(last_block)
        block_index += 1



class RacingMapHairpin(PGMap):
    """Tight U-turns with recovery straights. Tests handling and acceleration."""
    def _generate(self):
        assert len(self.road_network.graph) == 0, "Map is not empty, please create a new map to read config"
        _build_track(self, [
            ("straight", {Parameter.length: 100}),
            ("curve", {Parameter.length: 40, Parameter.radius: 25, Parameter.angle: 180, Parameter.dir: 1}),
            ("straight", {Parameter.length: 150}),
            ("curve", {Parameter.length: 40, Parameter.radius: 25, Parameter.angle: 180, Parameter.dir: 0}),
            ("straight", {Parameter.length: 150}),
            ("curve", {Parameter.length: 40, Parameter.radius: 30, Parameter.angle: 180, Parameter.dir: 1}),
            ("straight", {Parameter.length: 100}),
            ("curve", {Parameter.length: 60, Parameter.radius: 40, Parameter.angle: 90, Parameter.dir: 0}),
            ("straight", {Parameter.length: 80}),
        ])



RACING_MAPS = {
    "circuit": RacingMap,
    "hairpin": RacingMapHairpin,
}


def set_racing_map(map_name):
    """Monkey-patch the RacingMap class used by MultiAgentRacingEnv.

    Call this BEFORE creating the env. Returns a restore function to undo the patch.

    Usage:
        restore = set_racing_map("hairpin")
        env = MultiAgentRacingEnv(config)
        ...
        env.close()
        restore()
    """
    import metadrive.envs.marl_envs.marl_racing_env as _racing_mod
    if map_name not in RACING_MAPS:
        raise ValueError(f"Unknown map '{map_name}'. Available: {list(RACING_MAPS.keys())}")
    original = _racing_mod.RacingMap
    _racing_mod.RacingMap = RACING_MAPS[map_name]

    def restore():
        _racing_mod.RacingMap = original

    return restore
