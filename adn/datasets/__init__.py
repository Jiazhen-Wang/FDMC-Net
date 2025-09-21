import os.path as path
from .deep_lesion import DeepLesion
from .spineweb import Spineweb
from .nature_image import NatureImage
from .motion_image import MotionImage

def get_dataset(dataset_type, **dataset_opts):
    return {
        "deep_lesion": DeepLesion,
        "spineweb": Spineweb,
        "nature_image": NatureImage,
        "motion_image": MotionImage
    }[dataset_type](**dataset_opts[dataset_type])
