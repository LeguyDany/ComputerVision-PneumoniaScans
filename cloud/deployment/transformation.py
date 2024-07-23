import torchvision.transforms as tt


class DataTransformations:
    NORMALIZED_MEAN = [0.5, 0.5, 0.5]
    NORMALIZED_STD = [0.5, 0.5, 0.5]

    @classmethod
    def inference_transforms(cls, size:int):
        return tt.Compose([
            tt.Resize(size),
            tt.CenterCrop(size),
            tt.Grayscale(),
            tt.ToTensor(),
            lambda x: x.expand(3, -1, -1), # Transforms the grayscale into RGB
            tt.Normalize(cls.NORMALIZED_MEAN, cls.NORMALIZED_STD)
        ])