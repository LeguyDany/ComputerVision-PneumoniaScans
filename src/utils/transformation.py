import torchvision.transforms as tt

class DataTransformations:
    NORMALIZED_MEAN = [0.5, 0.5, 0.5]
    NORMALIZED_STD = [0.5, 0.5, 0.5]

    @classmethod
    def train_transforms(cls):
        return tt.Compose([
            tt.RandomHorizontalFlip(),
            tt.RandomRotation([-10, 10]),
            tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            tt.ToTensor(),
            tt.Normalize(cls.NORMALIZED_MEAN, cls.NORMALIZED_STD)
        ])
        

    @classmethod
    def valid_transforms(cls):
        return tt.Compose([
            tt.ToTensor(),
            tt.Normalize(cls.NORMALIZED_MEAN, cls.NORMALIZED_STD)
        ])
    

    @classmethod
    def test_transforms(cls):
        return tt.Compose([
            tt.ToTensor(),
            tt.Normalize(cls.NORMALIZED_MEAN, cls.NORMALIZED_STD)
        ])
