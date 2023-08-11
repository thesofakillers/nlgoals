from typing import Optional, Callable

from minigrid.wrappers import Wrapper


class ColorObjLockWrapper(Wrapper):
    """
    # TODO
    """

    def __init__(self, env, obj_type: str, color: str):
        super().__init__(env)
        self.obj_type = obj_type
        self.color = color
        self.wrapper_name = "color-obj-lock"
        raise NotImplementedError


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
