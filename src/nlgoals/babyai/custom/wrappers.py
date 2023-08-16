from typing import Optional, Callable
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel

from minigrid.wrappers import Wrapper

from nlgoals.babyai.custom.envs import (
    COLOR_NAMES,
    OBJ_MAP,
    CustomGoToColor,
    CustomGoToObj,
)


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

        self.env.tracked_type_positions = (
            {"key": [], "ball": [], "box": []} if self.track_types else None
        )
        self.env.tracked_color_positions = (
            {"red": [], "green": [], "blue": []} if self.track_colors else None
        )

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
            if self.env.tracked_type_positions is not None:
                self.env.tracked_type_positions[self.obj_type].append(
                    distractor.cur_pos
                )
        if self.color and distractor.color == self.color:
            color_count += 1
            if self.env.tracked_color_positions is not None:
                self.env.tracked_color_positions[self.color].append(distractor.cur_pos)
        return type_count, color_count

    def _ensure_minimums(self, type_count, color_count):
        if self.min_type:
            self._ensure_minimum(
                "type",
                self.min_type,
                self.obj_type,
                type_count,
                self.env.tracked_type_positions,
            )
        if self.min_color:
            self._ensure_minimum(
                "color",
                self.min_color,
                self.color,
                color_count,
                self.env.tracked_color_positions,
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
                            positions[getattr(distractor, attr)].append(
                                distractor.cur_pos
                            )
                    if count >= min_value:
                        break
