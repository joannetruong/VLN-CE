from typing import Any

import numpy as np
from gym import spaces
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


@registry.register_sensor(name="GlobalGPSSensor")
class GlobalGPSSensor(Sensor):
    r"""The agents current location in the global coordinate frame

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions
                to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "globalgps"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._dimensionality,),
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        return agent_position.astype(np.float32)


@registry.register_sensor
class VLNOracleActionSensor(Sensor):
    r"""Sensor for observing the optimal action to take. The assumption this
    sensor currently makes is that the shortest path to the goal is the
    optimal path.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)
        self.follower = ShortestPathFollower(
            self._sim,
            # all goals can be navigated to within 0.5m.
            goal_radius=getattr(config, "GOAL_RADIUS", 0.5),
            return_one_hot=False,
        )
        self.follower.mode = "geodesic_path"

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "vln_oracle_action_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        best_action = self.follower.get_next_action(episode.goals[0].position)
        return np.array(
            [best_action if best_action is not None else HabitatSimActions.STOP]
        )


@registry.register_sensor
class VLNOracleProgressSensor(Sensor):
    r"""Sensor for observing how much progress has been made towards the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "progress"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        # TODO: what is the correct sensor type?
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float)

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        distance_from_start = episode.info["geodesic_distance"]

        return (distance_from_start - distance_to_target) / distance_from_start
