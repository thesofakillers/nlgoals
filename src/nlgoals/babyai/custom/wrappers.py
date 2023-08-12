from typing import Optional, Callable
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel

from minigrid.wrappers import Wrapper


class ColorObjLockWrapper(Wrapper):
    """
    Ensures that a given object type will always be of a given color.
    """

    def __init__(self, env: RoomGridLevel, obj_type: str, color: str):
        """
        Args:
            env: The environment to wrap.
            obj_type: The object type to lock.
            color: The color to lock the object type to.
        """
        super().__init__(env)
        self.obj_type = obj_type
        self.color = color
        self.wrapper_name = "color-obj-lock"

    def reset(self, **kwargs):
        """
        Reset the environment and override object color.
        """
        obs = self.env.reset(**kwargs)

        # Override object color
        for obj in self.env.grid.grid:
            if obj is not None and obj.type == self.obj_type:
                obj.color = self.color

        return obs

class DistractorConstraintWrapper(Wrapper):
    """
    # TODO
    """

    def __init__(
        self,
        env,
        min_obj: Optional[int] = None,
        obj_type: Optional[str] = None,
        track_objects: Optional[bool] = False,
        min_color: Optional[int] = None,
        color: Optional[str] = None,
        track_colors: Optional[bool] = False,
    ):
        super().__init__(env)
        self.min_obj = min_obj
        self.obj_type = obj_type
        self.track_objects = track_objects
        self.min_color = min_color
        self.color = color
        self.track_colors = track_colors
        self.wrapper_name = "distractor-constraint"
        raise NotImplementedError
