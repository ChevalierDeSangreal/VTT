# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from aerial_gym.envs.base.aerial_robot_config import AerialRobotCfg
from .base.track_ground import TrackGroundVer7
from .base.track_ground_test import TrackGroundTest
from .base.track_ground_config import TrackGroundCfg
from .base.track_space_config import TrackSpaceCfg
from .base.track_space_config2 import TrackSpaceCfgVer2
from .base.track_agile_config import TrackAgileCfg
from .base.dynamics_isaac import IsaacGymDynamics, NRIsaacGymDynamics
from .base.dynamics_newton import NewtonDynamics
from .base.dynamics_simple import SimpleDynamics, NRSimpleDynamics
from .base.dynamics_isaac_origin import IsaacGymOriDynamics
from .base.track_space import TrackSpaceVer0
from .base.track_spaceVer2 import TrackSpaceVer2
from .base.track_spaceVer3 import TrackSpaceVer3
from .base.track_agileVer0 import TrackAgileVer0
from .base.track_agileVer1 import TrackAgileVer1
from .base.track_agileVer2 import TrackAgileVer2
from .base.track_agileVer3 import TrackAgileVer3
from aerial_gym.utils.task_registry import task_registry

# task_registry.register( "quad", AerialRobot, AerialRobotCfg())
# task_registry.register("quad_with_obstacles", AerialRobotWithObstacles, AerialRobotWithObstaclesCfg())
task_registry.register( "track_spaceVer0", TrackSpaceVer0, TrackSpaceCfg())
task_registry.register( "track_groundVer7", TrackGroundVer7, TrackGroundCfg())
task_registry.register( "track_ground_test", TrackGroundTest, TrackGroundCfg())
task_registry.register( "track_spaceVer2", TrackSpaceVer2, TrackSpaceCfgVer2())
task_registry.register( "track_spaceVer3", TrackSpaceVer3, TrackSpaceCfgVer2())
task_registry.register( "track_agileVer0", TrackAgileVer0, TrackSpaceCfgVer2())
task_registry.register( "track_agileVer1", TrackAgileVer1, TrackAgileCfg())
task_registry.register( "track_agileVer2", TrackAgileVer2, TrackAgileCfg())
task_registry.register( "track_agileVer3", TrackAgileVer3, TrackAgileCfg())