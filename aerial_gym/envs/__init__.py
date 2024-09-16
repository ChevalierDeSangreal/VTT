# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from aerial_gym.envs.base.aerial_robot_config import AerialRobotCfg
from .base.track_ground import TrackGroundVer7
from .base.track_ground_test import TrackGroundTest
from .base.track_ground_config import TrackGroundCfg
from .base.track_space_config import TrackSpaceCfg
from .base.dynamics_isaac import IsaacGymDynamics
from .base.dynamics_newton import NewtonDynamics
from .base.track_space import TrackSpaceVer0
from aerial_gym.utils.task_registry import task_registry

# task_registry.register( "quad", AerialRobot, AerialRobotCfg())
# task_registry.register("quad_with_obstacles", AerialRobotWithObstacles, AerialRobotWithObstaclesCfg())
task_registry.register( "track_spaceVer0", TrackSpaceVer0, TrackSpaceCfg())
task_registry.register( "track_groundVer7", TrackGroundVer7, TrackGroundCfg())
task_registry.register( "track_ground_test", TrackGroundTest, TrackGroundCfg())