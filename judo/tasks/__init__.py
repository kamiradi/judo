# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Dict, Type

from judo.tasks.base import Task, TaskConfig
from judo.tasks.caltech_leap_cube import CaltechLeapCube, CaltechLeapCubeConfig
from judo.tasks.cartpole import Cartpole, CartpoleConfig
from judo.tasks.cylinder_push import CylinderPush, CylinderPushConfig
from judo.tasks.fr3_pick import FR3Pick, FR3PickConfig
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig
from judo.tasks.leap_cube_down import LeapCubeDown, LeapCubeDownConfig
from judo.tasks.peg_insertion import PegInsertion, PegInsertionConfig
from judo.tasks.peg_insertion_twist import (
    PegInsertionTwist,
    PegInsertionTwistConfig,
    PegInsertionTwistEllipse,
    PegInsertionTwistEllipseTeeth,
    PegInsertionTwistRectangle,
    PegInsertionTwistRectangleTeeth,
)
from judo.tasks.spot import (
    SpotBase,
    SpotBaseConfig,
    SpotBoxPush,
    SpotBoxPushConfig,
    SpotNavigate,
    SpotNavigateConfig,
    SpotTireRoll,
    SpotTireRollConfig,
    SpotTireUpright,
    SpotTireUprightConfig,
)
from judo.tasks.spot.spot_constants import SPOT_LOCOMOTION_POLICY_PATH
from judo.tasks.uncertain_com_cylinder_push import UncertainComCylinderPush, UncertainComCylinderPushConfig
from judo.tasks.uncertain_cylinder_push import UncertainCylinderPush, UncertainCylinderPushConfig
from judo.tasks.uncertain_pos_com_cylinder_push import (
    UncertainPosComCylinderPush,
    UncertainPosComCylinderPushConfig,
)


@dataclass(frozen=True)
class TaskRegistration:
    """Complete registration metadata for a task."""

    task_type: Type[Task]
    task_config_type: Type[TaskConfig]
    rollout_backend: str = "mujoco"
    simulation_backend: str = "mujoco"
    locomotion_policy_path: str | None = None
    controller_cls: str | None = None
    """Optional import path of a Controller subclass to use for this task (e.g. a belief-aware controller)."""


_registered_tasks: Dict[str, TaskRegistration] = {
    CylinderPush.name: TaskRegistration(CylinderPush, CylinderPushConfig),
    UncertainCylinderPush.name: TaskRegistration(
        UncertainCylinderPush,
        UncertainCylinderPushConfig,
        controller_cls="judo.controller.belief_controller.BeliefAwareController",
    ),
    UncertainComCylinderPush.name: TaskRegistration(
        UncertainComCylinderPush,
        UncertainComCylinderPushConfig,
        controller_cls="judo.controller.param_belief_controller.ParamBeliefController",
    ),
    UncertainPosComCylinderPush.name: TaskRegistration(
        UncertainPosComCylinderPush,
        UncertainPosComCylinderPushConfig,
        controller_cls="judo.controller.pos_com_belief_controller.PosComBeliefController",
    ),
    Cartpole.name: TaskRegistration(Cartpole, CartpoleConfig),
    FR3Pick.name: TaskRegistration(FR3Pick, FR3PickConfig),
    LeapCube.name: TaskRegistration(LeapCube, LeapCubeConfig),
    LeapCubeDown.name: TaskRegistration(LeapCubeDown, LeapCubeDownConfig),
    CaltechLeapCube.name: TaskRegistration(CaltechLeapCube, CaltechLeapCubeConfig),
    PegInsertion.name: TaskRegistration(PegInsertion, PegInsertionConfig),
    PegInsertionTwist.name: TaskRegistration(PegInsertionTwist, PegInsertionTwistConfig),
    PegInsertionTwistRectangle.name: TaskRegistration(PegInsertionTwistRectangle, PegInsertionTwistConfig),
    PegInsertionTwistEllipse.name: TaskRegistration(PegInsertionTwistEllipse, PegInsertionTwistConfig),
    PegInsertionTwistEllipseTeeth.name: TaskRegistration(PegInsertionTwistEllipseTeeth, PegInsertionTwistConfig),
    PegInsertionTwistRectangleTeeth.name: TaskRegistration(PegInsertionTwistRectangleTeeth, PegInsertionTwistConfig),
    SpotBase.name: TaskRegistration(
        SpotBase,
        SpotBaseConfig,
        rollout_backend="mujoco_hierarchical",
        simulation_backend="mujoco_hierarchical",
        locomotion_policy_path=str(SPOT_LOCOMOTION_POLICY_PATH),
    ),
    SpotBoxPush.name: TaskRegistration(
        SpotBoxPush,
        SpotBoxPushConfig,
        rollout_backend="mujoco_hierarchical",
        simulation_backend="mujoco_hierarchical",
        locomotion_policy_path=str(SPOT_LOCOMOTION_POLICY_PATH),
    ),
    SpotNavigate.name: TaskRegistration(
        SpotNavigate,
        SpotNavigateConfig,
        rollout_backend="mujoco_hierarchical",
        simulation_backend="mujoco_hierarchical",
        locomotion_policy_path=str(SPOT_LOCOMOTION_POLICY_PATH),
    ),
    SpotTireRoll.name: TaskRegistration(
        SpotTireRoll,
        SpotTireRollConfig,
        rollout_backend="mujoco_hierarchical",
        simulation_backend="mujoco_hierarchical",
        locomotion_policy_path=str(SPOT_LOCOMOTION_POLICY_PATH),
    ),
    SpotTireUpright.name: TaskRegistration(
        SpotTireUpright,
        SpotTireUprightConfig,
        rollout_backend="mujoco_hierarchical",
        simulation_backend="mujoco_hierarchical",
        locomotion_policy_path=str(SPOT_LOCOMOTION_POLICY_PATH),
    ),
}


def get_registered_tasks() -> Dict[str, TaskRegistration]:
    """Returns a dictionary of registered tasks."""
    return _registered_tasks


def get_task_registration(task_name: str) -> TaskRegistration:
    """Return full registration metadata for a task."""
    task_entry = _registered_tasks.get(task_name)
    if task_entry is None:
        raise ValueError(f"Task {task_name} not found in task registry.")
    return task_entry


def register_task(
    name: str,
    task_type: Type[Task],
    task_config_type: Type[TaskConfig],
    rollout_backend: str = "mujoco",
    simulation_backend: str = "mujoco",
    locomotion_policy_path: str | None = None,
    controller_cls: str | None = None,
) -> None:
    """Registers a new task and its default controller/simulation backends."""
    _registered_tasks[name] = TaskRegistration(
        task_type=task_type,
        task_config_type=task_config_type,
        rollout_backend=rollout_backend,
        simulation_backend=simulation_backend,
        locomotion_policy_path=locomotion_policy_path,
        controller_cls=controller_cls,
    )


__all__ = [
    "get_registered_tasks",
    "get_task_registration",
    "register_task",
    "TaskRegistration",
    "Task",
    "TaskConfig",
    "CaltechLeapCube",
    "CaltechLeapCubeConfig",
    "Cartpole",
    "CartpoleConfig",
    "CylinderPush",
    "CylinderPushConfig",
    "FR3Pick",
    "FR3PickConfig",
    "LeapCube",
    "LeapCubeConfig",
    "LeapCubeDown",
    "LeapCubeDownConfig",
    "PegInsertion",
    "PegInsertionConfig",
    "PegInsertionTwist",
    "PegInsertionTwistConfig",
    "PegInsertionTwistEllipse",
    "PegInsertionTwistEllipseTeeth",
    "PegInsertionTwistRectangle",
    "PegInsertionTwistRectangleTeeth",
    "SpotBase",
    "SpotBaseConfig",
    "SpotBoxPush",
    "SpotBoxPushConfig",
    "SpotNavigate",
    "SpotNavigateConfig",
    "SpotTireRoll",
    "SpotTireRollConfig",
    "SpotTireUpright",
    "SpotTireUprightConfig",
    "UncertainComCylinderPush",
    "UncertainComCylinderPushConfig",
    "UncertainCylinderPush",
    "UncertainCylinderPushConfig",
    "UncertainPosComCylinderPush",
    "UncertainPosComCylinderPushConfig",
]
