import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataloader import DataLoader

from local_types.enums import EDataset, EOutput
from services.azure_bucket import AzureBucket
from utils.transformation import DataTransformations


class DataPreparer:
    def __init__(self, image_size: int = 75, train_ds=None, validation_ds=None, test_ds=None):
        self.azure_bucket = AzureBucket()
        self.image_size = image_size
        self.train_ds = train_ds
        self.validation_ds = validation_ds
        self.test_ds = test_ds
        self.fold_ds = None

    def __load_blob_from_local(self, dataset_path, dataset_type: EDataset):
        images = []
        labels = []

        path = f"{dataset_path}{dataset_type.name}"
        blobs = os.listdir(path)

        with tqdm(total=len(blobs), desc=f"Loading {dataset_type.name}", unit="file") as pbar:
            for blob in blobs:
                img = cv2.imread(f"{path}/{blob}")
                img = cv2.resize(img, (self.image_size, self.image_size))
                images.append(img)
                pbar.update(1)

                if "bacteria" in blob:
                    labels.append(EOutput.BACTERIA.value)
                    continue

                if "virus" in blob:
                    labels.append(EOutput.VIRUS.value)
                    continue

                labels.append(EOutput.NEGATIVE.value)

        return list(zip(images, labels))
    
    def __load_blob_from_container(self, dataset_type: EDataset):
        images = []
        labels = []
         
        blobs = self.azure_bucket.list_blobs(dataset_type)

        for blob in blobs:

            blob_client = self.azure_bucket.get_blob(dataset_type, blob.name)
            stream = blob_client.download_blob().readall()

            nparr = np.frombuffer(stream, np.uint8)

            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None: continue

            img = cv2.resize(img, (self.image_size, self.image_size))
            images.append(img)

            if "bacteria" in blob.name:
                labels.append(EOutput.BACTERIA.value)
                continue

            if "virus" in blob.name:
                labels.append(EOutput.VIRUS.value)
                continue

            labels.append(EOutput.NEGATIVE.value)

        return list(zip(images, labels))

    def load_datasets(self, locally: bool = False, dataset_path: str = f'{os.getcwd()}/dataset/'):
        print(f'{os.getcwd()}/dataset/')
        if(locally):
            self.train_ds = self.__load_blob_from_local(dataset_path, EDataset.TRAIN)
            self.validation_ds = self.__load_blob_from_local(dataset_path, EDataset.VALIDATION)
            self.test_ds = self.__load_blob_from_local(dataset_path, EDataset.TEST )
            self.fold_ds = self.train_ds + self.validation_ds
            return
        
        self.train_ds = self.__load_blob_from_container(EDataset.TRAIN)
        self.validation_ds = self.__load_blob_from_container(EDataset.VALIDATION)
        self.test_ds = self.__load_blob_from_container(EDataset.TEST)
        self.fold_ds = self.train_ds + self.validation_ds
        
    def get_dataloaders(self, dataset_type: EDataset, batch_size: int, shuffle: bool = False,  num_workers: int = 2, sampler=None):
        device = DeviceDataLoader.get_default_device()
        if dataset_type == EDataset.TRAIN:
            train_dl = [(DataTransformations.train_transforms()(Image.fromarray(image)), label) for image, label in self.train_ds]
            train_dl = DataLoader(train_dl, batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle)
            return DeviceDataLoader(train_dl, device)

        if dataset_type == EDataset.VALIDATION:
            validation_dl = [(DataTransformations.valid_transforms()(Image.fromarray(image)), label) for image, label in self.validation_ds]
            validation_dl = DataLoader(validation_dl, batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle)
            return DeviceDataLoader(validation_dl, device)

        if dataset_type == EDataset.FOLD:
            fold_dl = [(DataTransformations.train_transforms()(Image.fromarray(image)), label) for image, label in self.fold_ds]
            fold_dl = DataLoader(fold_dl, batch_size, num_workers=num_workers, pin_memory=True, sampler=sampler, shuffle=False)
            return DeviceDataLoader(fold_dl, device)

        test_dl = [(DataTransformations.test_transforms()(Image.fromarray(image)), label) for image, label in self.test_ds]
        test_dl = DataLoader(test_dl, batch_size, num_workers=num_workers, pin_memory=True)
        return DeviceDataLoader(test_dl, device)


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
