from models.SegmentationModel import SegmentationModel


def get_model(model_name):
    if model_name == "segmentation":
        return SegmentationModel()
