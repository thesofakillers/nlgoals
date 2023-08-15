from typing import Optional, Callable
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel

from minigrid.wrappers import Wrapper

from nlgoals.babyai.custom.envs import (
    COLOR_NAMES,
    OBJ_MAP,
    CustomGoToColor,
    CustomGoToObj,
)


class ColorTypeLockWrapper(Wrapper):
    """
    Ensures that a given object type will always be of a given color.
    """

    def __init__(
        self,
        env: RoomGridLevel,
        obj_type: str,
        color: str,
        track_types: bool = False,
        track_colors: bool = False,
    ):
        """
        Args:
            env: The environment to wrap.
            obj_type: The object type to lock.
            color: The color to lock the object type to.
            track_types: track the positions of objects of type `obj_type` in the env.
                in self.tracked_type_positions.
            track_colors: track the positions of objects of color `color` in the env.
                in self.tracked_color_positions.
        """
        super().__init__(env)
        self.obj_type = obj_type
        self.color = color
        self.wrapper_name = "color-obj-lock"
        self.track_types = track_types
        self.track_colors = track_colors

    def reset(self, **kwargs):
        """
        Reset the environment and override object color.
        """
        obs = self.env.reset(**kwargs)

        goal_obj = self.env.unwrapped.goal_obj
        only_one = self.env.only_one

        if goal_obj.color != self.color and isinstance(self.env, CustomGoToColor):
            if goal_obj.type == self.obj_type:
                raise ValueError("Cannot lock color of goal in GoToColor env.")

        if goal_obj.type != self.obj_type and isinstance(self.env, CustomGoToObj):
            if goal_obj.color == self.color:
                raise ValueError("Cannot lock type of goal in GoToObj env.")

        if self.track_types:
            self.tracked_type_positions = []
        if self.track_colors:
            self.tracked_color_positions = []

        type_color_count = 0
        # for object in grid, if type, make color, if color make type
        # if only_one and the goal related to our wrapper,
        # do this once and change the others to the other type/color
        for obj in self.env.grid.grid:
            if obj is None:
                continue
            if (
                only_one
                and type_color_count >= 1
                and (goal_obj.type == self.obj_type or goal_obj.color == self.color)
            ):
                if obj.type == self.obj_type:
                    obj.type = self._rand_elem(
                        set(OBJ_MAP.keys()) - set([self.obj_type])
                    )
                if obj.color == self.color:
                    obj.color = self._rand_elem(set(COLOR_NAMES) - set([self.color]))
            elif obj.type == self.obj_type:
                obj.color = self.color
                type_color_count += 1
            elif obj.color == self.color:
                obj.type = self.obj_type
                type_color_count += 1
            if obj.type == self.obj_type and self.track_types:
                self.tracked_type_positions.append(obj.cur_pos)
            if obj.color == self.color and self.track_colors:
                self.tracked_color_positions.append(obj.cur_pos)

        return obs


class DistractorConstraintWrapper(Wrapper):
    """
    Ensures that there are at either
    - at least `min_type` distractors of type `obj_type` in the environment
    - at least `min_color` distractors of color `color` in the environment
    """

    def __init__(
        self,
        env,
        min_type=None,
        obj_type=None,
        track_types=False,
        min_color=None,
        color=None,
        track_colors=False,
    ):
        """
        Args:
            env: The environment to wrap.
            min_type: min num of distractors of type `obj_type` that must be in the env.
            obj_type: The object type to count.
            track_types: track the positions of objects of type `obj_type` in the env.
                in self.tracked_type_positions.
            min_color: min num of distractors of color `color` that must be in the env.
            color: The color to count.
            track_colors: track the positions of objects of color `color` in the env.
                in self.tracked_color_positions.

        """
        super().__init__(env)
        self.min_type = min_type
        self.obj_type = obj_type
        self.track_types = track_types
        self.min_color = min_color
        self.color = color
        self.track_colors = track_colors
        self.wrapper_name = "distractor-constraint"

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self.tracked_type_positions = [] if self.track_types else None
        self.tracked_color_positions = [] if self.track_colors else None

        type_count, color_count = self._count_and_track()
        self._ensure_minimums(type_count, color_count)

        return obs

    def _count_and_track(self):
        type_count = color_count = 0
        for distractor in self.unwrapped.distractors:
            type_count, color_count = self._update_counts_and_positions(
                distractor, type_count, color_count
            )
        return type_count, color_count

    def _update_counts_and_positions(self, distractor, type_count, color_count):
        if self.obj_type and distractor.type == self.obj_type:
            type_count += 1
            if self.tracked_type_positions is not None:
                self.tracked_type_positions.append(distractor.cur_pos)
        if self.color and distractor.color == self.color:
            color_count += 1
            if self.tracked_color_positions is not None:
                self.tracked_color_positions.append(distractor.cur_pos)
        return type_count, color_count

    def _ensure_minimums(self, type_count, color_count):
        if self.min_type:
            self._ensure_minimum(
                "type",
                self.min_type,
                self.obj_type,
                type_count,
                self.tracked_type_positions,
            )
        if self.min_color:
            self._ensure_minimum(
                "color",
                self.min_color,
                self.color,
                color_count,
                self.tracked_color_positions,
            )

    def _ensure_minimum(self, attr, min_value, attr_value, count, positions):
        """
        If the number of distractors of `attr`, `attr_value` is less than `min_value`,
        tries to replace other distractors with distractors of `attr`, `attr_value`.

        Args:
            attr: "type" or "color"
            min_value: The minimum number of distractors of `attr`, `attr_value`
            attr_value: The value of `attr` to count.
            count: The number of distractors of `attr`, `attr_value` in the environment.
            positions: The positions of distractors of `attr`, `attr_value` in the environment.
                either self.tracked_type_positions or self.tracked_color_positions.
        """
        if count < min_value:
            if len(self.unwrapped.distractors) < (min_value - count):
                raise ValueError(
                    f"Cannot ensure minimum {min_value} of {attr} {attr_value}. Not enough distractors."
                )
            else:
                for distractor in self.unwrapped.distractors:
                    if getattr(distractor, attr) != attr_value:
                        setattr(distractor, attr, attr_value)
                        count += 1
                        if positions is not None:
                            positions.append(distractor.cur_pos)
                    if count >= min_value:
                        break
