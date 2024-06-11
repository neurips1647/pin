#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import TYPE_CHECKING, Any, List, Optional

import attr
import numpy as np
from gym import spaces
import magnum

import random
import magnum as mn
import math

import habitat_sim
from habitat.core.dataset import Dataset
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes, Simulator
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True, kw_only=True)
class MultiObjectGoalNavEpisode(NavigationEpisode):
    r"""Multi ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[List[str]] = None
    object_index: Optional[int]
    current_goal_index: Optional[int] = 0
    distractors: List[Any] = []

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return [
            f"{os.path.basename(self.scene_id)}_{i}"
            for i in self.object_category
        ]


@attr.s(auto_attribs=True)
class ObjectViewLocation:
    r"""ObjectViewLocation provides information about a position around an object goal
    usually that is navigable and the object is visible with specific agent
    configuration that episode's dataset was created.
     that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        agent_state: navigable AgentState with a position and a rotation where
        the object is visible.
        iou: an intersection of a union of the object and a rectangle in the
        center of view. This metric is used to evaluate how good is the object
        view form current position. Higher iou means better view, iou equals
        1.0 if whole object is inside of the rectangle and no pixel inside
        the rectangle belongs to anything except the object.
    """
    agent_state: AgentState
    iou: Optional[float]


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_name: Optional[str] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[ObjectViewLocation]] = None


@attr.s(auto_attribs=True, kw_only=True)
class MultiObjectGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.
    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    position: Optional[List[List[float]]]
    object_id: Optional[str] = None
    viewpoints: Optional[list] = None


@registry.register_sensor
class MultiObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            goal_spec that specifies which id use for goal specification,
            goal_spec_max_val the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """

    def __init__(
        self,
        sim,
        config: "DictConfig",
        dataset: Dataset,
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "multiobjectgoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = (self.config.goal_spec_max_val - 1,)
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: MultiObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[int]:
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            if len(episode.goals) == 0:
                logger.error(
                    f"No goal specified for episode {episode.episode_id}."
                )
                return None
            category_name = [i.object_category for i in episode.goals]
            goalArray = np.array(
                [
                    self._dataset.category_to_task_category_id[i]
                    for i in category_name
                ],
                dtype=np.int64,
            )
            return goalArray[
                kwargs["task"]
                .current_goal_index : kwargs["task"]
                .current_goal_index
                + 1
            ]
        elif self.config.goal_spec == "OBJECT_ID":
            return np.array([episode.goals[0].object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong goal_spec specified for ObjectGoalSensor."
            )


@registry.register_task(name="MultiObjectNav-v1")
class MultiObjectNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """

    def __init__(
        self,
        config: "DictConfig",
        sim: Simulator,
        dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
        self.current_goal_index = 0
        self.object_to_datset_mapping = dataset.category_to_task_category_id

    def reset(self, episode: MultiObjectGoalNavEpisode):
        rigid_obj_mgr = self._sim.get_rigid_object_manager()

        # Remove existing objects from last episode
        rigid_obj_mgr.remove_all_objects()

        # Insert current episode objects
        obj_path = self._config.objects_path

        obj_templates_mgr = self._sim.get_object_template_manager()
        obj_templates_mgr.load_configs(obj_path, True)

        for i in range(len(episode.goals)):
            current_goal = episode.goals[i].object_category
            object_id = episode.goals[i].object_id
            dataset_index = self.object_to_datset_mapping[current_goal]

            # obj_handle_list = obj_templates_mgr.get_template_handles(object_id)[0]
            obj_handle_list = obj_templates_mgr.get_template_handles(
                object_id
            )[0]
            object_box = rigid_obj_mgr.add_object_by_template_handle(
                obj_handle_list
            )
            obj_node = object_box.root_scene_node
            obj_bb = obj_node.cumulative_bb
            jj = obj_bb.back_bottom_left
            jj = [jj[0], jj[2], jj[1]]
            diff = np.array(episode.goals[i].position)
            diff2 = diff - jj
            diff2[2] += jj[2] * 2
            diff2[1] += 0.05
            object_box.semantic_id = dataset_index
            object_box.translation = np.array(diff2)
            object_box.rotate_x(magnum.Rad(-1.5708))

        if self._config.include_distractors:
            for i in range(len(episode.distractors)):
                current_distractor = episode.distractors[i].object_category
                object_id = episode.distractors[i].object_id

                dataset_index = self.object_to_datset_mapping[
                    current_distractor
                ]
                try:
                    obj_handle_list = obj_templates_mgr.get_template_handles(
                        object_id
                    )[0]
                    object_box = rigid_obj_mgr.add_object_by_template_handle(
                        obj_handle_list
                    )
                    obj_node = object_box.root_scene_node
                    obj_bb = obj_node.cumulative_bb
                    jj = obj_bb.back_bottom_left
                    jj = [jj[0], jj[2], jj[1]]
                    diff = np.array(episode.distractors[i].position)
                    diff2 = diff - jj
                    diff2[2] += jj[2] * 2
                    diff2[1] += 0.05
                    object_box.semantic_id = dataset_index
                    object_box.translation = np.array(diff2)
                    object_box.rotate_x(magnum.Rad(-1.5708))
                except:
                    with open("logs/multion_missing_distractors.txt", "a") as f:
                        f.write(f"{object_id}\n")

        # Reinitialize current goal index
        self.current_goal_index = 0

        # Initialize self.is_found_called
        self.is_found_called = False

        observations = self._sim.reset()
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations, episode=episode, task=self
            )
        )

        for action_instance in self.actions.values():
            action_instance.reset(episode=episode, task=self)

        return observations
