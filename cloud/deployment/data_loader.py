import io
import torch
from PIL import Image

from transformation import DataTransformations


class InferenceDataPreparer:
    def __init__(self, image_blob, image_size: int = 75):
        self.image_size = image_size
        self.image_loaded = self.__load_blob_from_file(image_blob)

    def __load_blob_from_file(self, image_blob):
        image = Image.open(io.BytesIO(image_blob))
        image = DataTransformations.inference_transforms(self.image_size)(image)
        image = image.unsqueeze(0)
        image = image.to(DeviceDataLoader.get_default_device())
        return image


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device = None):
        self.dl = dl
        if(device is None):
            self.device = self.get_default_device()
        else:
            self.device = device

    @staticmethod
    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    @classmethod
    def to_device(cls, data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list,tuple)):
            return [cls.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield self.to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
