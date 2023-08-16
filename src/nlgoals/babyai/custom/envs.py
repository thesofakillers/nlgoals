"""Custom Environments for Minigrid"""
import math
from typing import Optional, List, Tuple, Set

from minigrid.core.constants import COLOR_TO_IDX
from minigrid.core.grid import OBJECT_TO_IDX
from minigrid.core.roomgrid import Ball, Box, Key, Room
from minigrid.core.world_object import WorldObj, Point
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc
import numpy as np

POSSIBLE_CC_POS = {"top left", "top right", "bottom left", "bottom right"}

OBJ_MAP = {
    "key": Key,
    "ball": Ball,
    "box": Box,
}
OBJ_NAMES = {"key", "ball", "box"}
COLOR_MAP = {
    "red": {},
    "green": {},
    "blue": {},
}
COLOR_NAMES = {"red", "green", "blue"}


def get_obj(obj_type, obj_color):
    return OBJ_MAP[obj_type](obj_color)


class CustomGoToObj(RoomGridLevel):
    """
    "Go to the/a {obj_type}" task
    """

    def __init__(
        self,
        obj_type: Optional[str] = None,
        only_one: bool = False,
        num_dists: Optional[int] = None,
        not_colors: Optional[Set[str]] = None,
        color_to_type: Optional[dict] = None,
        track_colors: bool = False,
        track_types: bool = False,
        **kwargs,
    ):
        """
        Args:
            obj_type: type of object to go to. If not specified, will be random
            only_one: whether to have only one object of the given type
            num_dists: number of distractors to place
            not_colors: set of colors to exclude from possible goal object colors
            color_to_type: Mapping to lock color and object type exclusively
            track_colors: whether to track the positions of colors
            track_types: whether to track the positions of types
        """
        if obj_type is None:
            self.obj_type = self._rand_elem(OBJ_MAP.keys())
        else:
            assert (
                obj_type in OBJ_MAP
            ), f"Invalid object type: {obj_type}. Must be one of {OBJ_MAP.keys()}."
            self.obj_type = obj_type
        self.only_one = only_one
        if num_dists is None:
            num_dists = self._rand_int(0, 8)
        self.num_dists = num_dists
        self.not_colors = not_colors
        self.color_to_type = color_to_type
        self.type_to_color = (
            {v: k for k, v in color_to_type.items()} if color_to_type else None
        )
        self.track_colors = track_colors
        self.track_types = track_types

        super().__init__(num_rows=1, num_cols=1, room_size=8, **kwargs)

    def gen_mission(self):
        # randomly place agent
        self.place_agent()

        # define goal object
        self.obj_color = self.get_obj_color(self.obj_type, self.get_goal_color)
        self.goal_obj = get_obj(self.obj_type, self.obj_color)

        # randomly place obj
        self.place_obj(self.goal_obj)

        # distractors
        self.distractor_types = set(OBJ_MAP.keys())
        if self.only_one:
            self.distractor_types -= {self.obj_type}
        self.place_distractors()

        self.check_objs_reachable()

        # generate goal/mission
        self.instrs = GoToInstr(ObjDesc(type=self.obj_type))

        self.grid_iterate_checks()

    def grid_iterate_checks(self):
        if self.track_colors:
            self.tracked_color_positions = {"red": [], "green": [], "blue": []}
        if self.track_types:
            self.tracked_type_positions = {"key": [], "ball": [], "box": []}
        for obj in self.grid.grid:
            if obj is None:
                continue
            if self.track_types and obj.type in self.tracked_type_positions:
                self.tracked_type_positions[obj.type].append(obj.cur_pos)
            if self.track_colors and obj.color in self.tracked_color_positions:
                self.tracked_color_positions[obj.color].append(obj.cur_pos)
            if self.color_to_type is not None:
                if obj.color in self.color_to_type:
                    assert (
                        obj.type == self.color_to_type[obj.color]
                    ), "Something went wrong"
                if obj.type in self.type_to_color:
                    assert (
                        obj.color == self.type_to_color[obj.type]
                    ), "Something went wrong"

    def get_goal_color(self, color_names):
        if self.not_colors is None:
            color = self._rand_elem(color_names)
        else:
            color = self._rand_elem(color_names - self.not_colors)
        return color

    def get_obj_color(self, obj_type, alt_func):
        if self.type_to_color is not None:
            if obj_type in self.type_to_color:
                obj_color = self.type_to_color[obj_type]
            else:
                obj_color = self._rand_elem(
                    COLOR_NAMES - set(self.color_to_type.keys())
                )
        else:
            obj_color = alt_func(COLOR_NAMES)

        return obj_color

    def place_distractors(self):
        self.distractors = []
        for _ in range(self.num_dists):
            distractor_type = self._rand_elem(self.distractor_types)
            distractor_color = self.get_obj_color(distractor_type, self._rand_elem)
            distractor = get_obj(distractor_type, distractor_color)
            self.place_obj(distractor)
            self.distractors.append(distractor)


class CustomGoToColor(RoomGridLevel):
    """
    "Go to the/a {obj_color} object" task
    """

    def __init__(
        self,
        obj_color: Optional[str] = None,
        only_one: bool = False,
        num_dists: Optional[int] = None,
        not_types: Optional[Set[str]] = None,
        color_to_type: Optional[dict] = None,
        track_colors: bool = False,
        track_types: bool = False,
        **kwargs,
    ):
        """
        Args:
            obj_color: color of object to go to. If not specified, will be random
            only_one: whether to have only one object of the given type
            num_dists: number of distractors to place
            not_types: set of types to exclude from possible goal object colors
            color_to_type: Mapping to lock color and object type exclusively
            track_colors: whether to track the positions of colors
            track_types: whether to track the positions of types
        """
        if obj_color is None:
            self.obj_color = self._rand_elem(COLOR_MAP.keys())
        else:
            assert (
                obj_color in COLOR_MAP
            ), f"Invalid object color: {obj_color}. Must be one of {COLOR_MAP.keys()}."
            self.obj_color = obj_color
        self.only_one = only_one
        if num_dists is None:
            num_dists = self._rand_int(0, 8)
        self.num_dists = num_dists
        self.not_types = not_types
        self.color_to_type = color_to_type
        self.type_to_color = (
            {v: k for k, v in color_to_type.items()} if color_to_type else None
        )
        self.track_colors = track_colors
        self.track_types = track_types

        super().__init__(num_rows=1, num_cols=1, room_size=8, **kwargs)

    def gen_mission(self):
        self.place_agent()

        # define goal object
        self.obj_type = self.get_obj_type(self.obj_color, self.get_goal_type)
        self.goal_obj = get_obj(self.obj_type, self.obj_color)

        # randomply place obj
        self.place_obj(self.goal_obj)

        # distractors
        self.distractor_colors = set(COLOR_MAP.keys())
        if self.only_one:
            self.distractor_colors -= {self.obj_color}
        self.place_distractors()

        self.check_objs_reachable()

        # generate goal/mission
        self.instrs = GoToInstr(ObjDesc(type=None, color=self.obj_color))

        self.grid_iterate_checks()

    def grid_iterate_checks(self):
        if self.track_colors:
            self.tracked_color_positions = {"red": [], "green": [], "blue": []}
        if self.track_types:
            self.tracked_type_positions = {"key": [], "ball": [], "box": []}
        for obj in self.grid.grid:
            if obj is None:
                continue
            if self.track_types and obj.type in self.tracked_type_positions:
                self.tracked_type_positions[obj.type].append(obj.cur_pos)
            if self.track_colors and obj.color in self.tracked_color_positions:
                self.tracked_color_positions[obj.color].append(obj.cur_pos)
            if self.color_to_type is not None:
                if obj.color in self.color_to_type:
                    assert (
                        obj.type == self.color_to_type[obj.color]
                    ), "Something went wrong"
                if obj.type in self.type_to_color:
                    assert (
                        obj.color == self.type_to_color[obj.type]
                    ), "Something went wrong"

    def get_goal_type(self, type_names):
        if self.not_types is None:
            obj_type = self._rand_elem(type_names)
        else:
            obj_type = self._rand_elem(type_names - self.not_types)
        return obj_type

    def get_obj_type(self, obj_color, alt_func):
        if self.color_to_type is not None:
            if obj_color in self.color_to_type:
                obj_type = self.color_to_type[obj_color]
            else:
                obj_type = self._rand_elem(OBJ_NAMES - set(self.type_to_color.keys()))
        else:
            obj_type = alt_func(OBJ_NAMES)

        return obj_type

    def place_distractors(self):
        self.distractors = []

        for _ in range(self.num_dists):
            distractor_color = self._rand_elem(self.distractor_colors)
            distractor_type = self.get_obj_type(distractor_color, self._rand_elem)
            distractor = get_obj(distractor_type, distractor_color)
            self.place_obj(distractor)
            self.distractors.append(distractor)


# class CustomGoToColor(RoomGridLevel):
#     """
#     "Go to the/a {obj_color} object" task
#     """

#     def __init__(
#         self,
#         obj_color: Optional[str] = None,
#         only_one: bool = False,
#         num_dists: Optional[int] = None,
#         **kwargs,
#     ):
#         """
#         Args:
#             obj_color: color of object to go to. If not specified, will be random
#             only_one: whether to have only one object of the given color
#             num_dists: number of distractors to place
#         """
#         if obj_color is None:
#             self.obj_color = self._rand_elem(COLOR_NAMES)
#         else:
#             assert (
#                 obj_color in COLOR_NAMES
#             ), f"Invalid object color: {obj_color}. Must be one of {COLOR_NAMES}."
#             self.obj_color = obj_color
#         self.only_one = only_one
#         if num_dists is None:
#             num_dists = self._rand_int(0, 8)
#         self.num_dists = num_dists
#         super().__init__(num_rows=1, num_cols=1, room_size=8, **kwargs)

#     def gen_mission(self):
#         # randomly place agent
#         self.place_agent()

#         # setup object
#         #   randomly choose object type
#         self.set_obj_type()
#         #   init obj instance
#         self.goal_obj = get_obj(self.obj_type, self.obj_color)
#         # randomly place obj
#         self.place_obj(self.goal_obj)

#         # distractors
#         self.distractor_colors = set(COLOR_NAMES)
#         if self.only_one:
#             self.distractor_colors -= {self.obj_color}
#         self.place_distractors()

#         self.check_objs_reachable()

#         # generate goal/mission
#         self.instrs = GoToInstr(ObjDesc(type=None, color=self.obj_color))

#     def set_obj_type(self):
#         self.obj_type = self._rand_elem(OBJ_MAP.keys())

#     def place_distractors(self):
#         self.distractors = []
#         for _ in range(self.num_dists):
#             distractor_type = self._rand_elem(OBJ_MAP.keys())
#             distractor_color = self._rand_elem(self.distractor_colors)
#             distractor = get_obj(distractor_type, distractor_color)
#             self.place_obj(distractor)
#             self.distractors.append(distractor)


class RoomGridLevelCC(RoomGridLevel):
    """
    Modification of RoomGridLevel in which a certain object is causally confused (CC)

    RoomGridLevel inherits from RoomGrid, which inherits from MiniGridEnv, which
    implements the place_obj method. We override this method s.t. that the causally
    confused object is always placed in the same location. This is to causally confuse
    the object with the location.

    We also override the `add_distractors` of RoomGrid method to make sure that if the
    CC object is used as a distractor, there is only one instance of it and it is placed
    in the same location.

    RoomGridLevelCC should then be used as the base class for any RoomGridLevel that we
    wish to have a CC object in.

    Note: Only tested for small envs.

    e.g.

    ```python
    from minigrid.envs.babyai import GoToLocal
    from nlgoals.babyai.custom import RoomGridLevelCC
    GoToLocal.__bases__ = (RoomGridLevelCC,)

    env = GoToLocal(cc_obj_kind="key", cc_obj_color="red", cc_obj_pos="top left")
    ```
    """

    def __init__(
        self, cc_obj_kind: str, cc_obj_color: str, cc_obj_pos_str: str, **kwargs
    ):
        """
        Args:
            cc_obj_kind: The kind of object to causally confuse
            cc_obj_color: The color of the object to causally confuse.
            cc_obj_pos_str: The position of the object to causally confuse. Must be one of
                {"top left", "top right", "bottom left", "bottom right"}
        """
        super().__init__(**kwargs)
        assert (
            cc_obj_kind in OBJECT_TO_IDX
        ), f"Invalid object kind: {cc_obj_kind}. Must be one of {OBJECT_TO_IDX.keys()}."
        assert (
            cc_obj_color in COLOR_TO_IDX
        ), f"Invalid object color: {cc_obj_color}. Must be one of {COLOR_TO_IDX.keys()}."
        assert (
            cc_obj_pos_str in POSSIBLE_CC_POS
        ), f"Invalid cc_obj_pos: {cc_obj_pos_str}. Must be one of {POSSIBLE_CC_POS}."

        self.cc_obj_kind = cc_obj_kind
        self.cc_obj_kind_idx = OBJECT_TO_IDX[cc_obj_kind]

        self.cc_obj_color = cc_obj_color
        self.cc_obj_color_idx = COLOR_TO_IDX[cc_obj_color]

        self.cc_obj = WorldObj.decode(self.cc_obj_kind_idx, self.cc_obj_color_idx, 0)

        self.cc_obj_pos_str = cc_obj_pos_str

    def add_distractors(
        self,
        i: Optional[int] = None,
        j: Optional[int] = None,
        num_distractors: int = 10,
        all_unique: bool = True,
        cc_obj_instances: int = 0,
    ) -> List[WorldObj]:
        """
        Add random objects that can potentially distract/confuse the agent.

        Ensures that the causally confused object is added at most once.
        self.place_obj takes care of making sure that the causally confused object is
        always placed in the same location, if placed.
        """

        # Collect a list of existing objects
        objs = []
        for row in self.room_grid:
            for room in row:
                for obj in room.objs:
                    objs.append((obj.type, obj.color))

        # List of distractors added
        dists = []

        while len(dists) < num_distractors:
            color = self._rand_elem(COLOR_NAMES)
            type = self._rand_elem(["key", "ball", "box"])

            # only add one instance of the cc object
            if color == self.cc_obj_color and type == self.cc_obj_kind:
                if cc_obj_instances > 0:
                    continue
                else:
                    cc_obj_instances += 1

            obj = (type, color)

            if all_unique and obj in objs:
                continue

            # Add the object to a random room if no room specified
            room_i = i
            room_j = j
            if room_i is None:
                room_i = self._rand_int(0, self.num_cols)
            if room_j is None:
                room_j = self._rand_int(0, self.num_rows)

            dist, pos = self.add_object(room_i, room_j, *obj)

            objs.append(obj)
            dists.append(dist)

        return dists

    def place_obj(
        self,
        obj: Optional[WorldObj],
        top: Point = None,
        size: Tuple[int, int] = None,
        reject_fn=None,
        max_tries=math.inf,
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        left_pos = top[0] + 1
        right_pos = top[0] + size[0] - 2
        top_pos = top[1] + 1
        bottom_pos = top[1] + size[1] - 2
        # -2 to account for wall width
        possible_cc_obj_pos = {
            "top left": (left_pos, top_pos),
            "top right": (right_pos, top_pos),
            "bottom left": (left_pos, bottom_pos),
            "bottom right": (right_pos, bottom_pos),
        }
        self.cc_obj_pos = possible_cc_obj_pos[self.cc_obj_pos_str]

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            if (
                obj is not None
                and obj.type == self.cc_obj_kind
                and obj.color == self.cc_obj_color
            ):
                pos = self.cc_obj_pos
            else:
                pos = (
                    self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                    self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
                )
                # don't place an obj where the cc obj should/could be
                if pos == self.cc_obj_pos:
                    continue

            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos


class GoToSpecObj(RoomGridLevel):
    """
    ## Description

    Go to the {color} {type}, single room, with distractors.

    Reimplementation of minigrid.envs.babyai.goto.GoToRedBall, such that all objects in
    the room are unique, so that there is a single spec obj to go to.
    """

    def __init__(
        self, obj_kind="ball", obj_color="red", room_size=8, num_dists=7, **kwargs
    ):
        self.num_dists = num_dists
        assert obj_kind in {"ball", "key", "box"}, "invalid obj_type"
        self.obj_kind = obj_kind
        assert obj_color in COLOR_NAMES, "invalid obj_color"
        self.obj_color = obj_color
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, self.obj_kind, self.obj_color)
        add_distractors_kwargs = {
            "num_distractors": self.num_dists,
            "all_unique": True,
        }
        if self.__class__.__bases__[0] == RoomGridLevelCC:
            add_distractors_kwargs["cc_obj_instances"] = 1
        self.add_distractors(**add_distractors_kwargs)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))
