from models.ClassificationModel import ClassificationModel
from models.SegmentationModel import SegmentationModel


def get_model(model_name):
    if model_name == "classification":
        return ClassificationModel()
    elif model_name == "segmentation":
        return SegmentationModel()
