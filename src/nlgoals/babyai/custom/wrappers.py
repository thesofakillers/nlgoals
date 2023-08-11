from typing import Optional, Callable

from minigrid.wrappers import Wrapper


class ColorObjLockWrapper(Wrapper):
    """
    # TODO
    """

    def __init__(self, obj_type: str, color: str, **kwargs):
        self.obj_type = obj_type
        self.color = color
        super().__init__(**kwargs)
        raise NotImplementedError


class DistractorConstraintWrapper(Wrapper):
    """
    # TODO
    """

    def __init__(
        self,
        min_obj: Optional[int] = None,
        obj_type: Optional[str] = None,
        min_color: Optional[int] = None,
        color: Optional[str] = None,
        **kwargs,
    ):
        self.min_obj = min_obj
        self.obj_type = obj_type
        self.min_color = min_color
        self.color = color
        super().__init__(**kwargs)
        raise NotImplementedError


class RGBImgTransformWrapper(Wrapper):
    """
    # TODO
    """

    def __init__(self, img_transform: Callable, **kwargs):
        self.img_transform = img_transform
        super().__init__(**kwargs)
        raise NotImplementedError


class MissionTransformWrapper(Wrapper):
    """
    # TODO
    """

    def __init__(self, mission_transform: Callable, **kwargs):
        self.mission_transform = mission_transform
        super().__init__(**kwargs)
        raise NotImplementedError
