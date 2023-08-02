"""Custom Environments for Minigrid"""
import math
from typing import Optional, List, Tuple

from minigrid.core.constants import COLOR_NAMES, COLOR_TO_IDX
from minigrid.core.grid import OBJECT_TO_IDX
from minigrid.core.world_object import WorldObj, Point
from minigrid.core.roomgrid import RoomGrid
import numpy as np

POSSIBLE_CC_POS = {"top left", "top right", "bottom left", "bottom right"}


class RoomGridCC(RoomGrid):
    """
    Modification of RoomGrid in which a certain object is causally confused (CC)

    RoomGrid inherits from MiniGridEnv, which implements the place_obj method. We
    override this method s.t. that the causally confused object is always placed in the
    same location. This is to causally confuse the object with the location.

    We also override the `add_distractors` method to make sure that if the CC object is
    used as a distractor, there is only one instance of it and it is placed in the same
    location.
    """

    def __init__(self, cc_obj_kind: str, cc_obj_color: str, cc_obj_pos: str, **kwargs):
        super().__init__(**kwargs)
        assert (
            cc_obj_kind in OBJECT_TO_IDX
        ), f"Invalid object kind: {cc_obj_kind}. Must be one of {OBJECT_TO_IDX.keys()}."
        assert (
            cc_obj_color in COLOR_TO_IDX
        ), f"Invalid object color: {cc_obj_color}. Must be one of {COLOR_TO_IDX.keys()}."
        assert (
            cc_obj_pos in POSSIBLE_CC_POS
        ), f"Invalid cc_obj_pos: {cc_obj_pos}. Must be one of {POSSIBLE_CC_POS}."

        self.cc_obj_kind = cc_obj_kind
        self.cc_obj_kind_idx = OBJECT_TO_IDX[cc_obj_kind]

        self.cc_obj_color = cc_obj_color
        self.cc_obj_color_idx = COLOR_TO_IDX[cc_obj_color]

        self.cc_obj = WorldObj.decode(self.cc_obj_kind_idx, self.cc_obj_color_idx, 0)

    def add_distractors(
        self,
        i: Optional[int] = None,
        j: Optional[int] = None,
        num_distractors: int = 10,
        all_unique: bool = True,
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

        cc_obj_instances = 0

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

        # x, y
        possible_cc_obj_pos = {
            "top left": top,
            "rop right": (top[0] + size[0] - 1, top[1]),
            "bottom left": (top[0], top[1] + size[1] - 1),
            "bottom right": (top[0] + size[0] - 1, top[1] + size[1] - 1),
        }
        cc_obj_pos = possible_cc_obj_pos[self.cc_obj_pos]

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
                pos = cc_obj_pos
            else:
                pos = (
                    self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                    self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
                )
                # don't place an obj where the cc obj should/could be
                if pos == cc_obj_pos:
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
