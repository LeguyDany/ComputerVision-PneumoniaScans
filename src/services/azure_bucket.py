from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from local_types.enums import EDataset
from tqdm import tqdm
import os
from dotenv import load_dotenv


class AzureBucket:
    load_dotenv()
    CONNECTION_STRING = os.getenv('AZURE_CONNECTION_STRING')
    CONTAINER_TRAIN_NAME = "epitech-zoidberg-dataset-train"
    CONTAINER_VAL_NAME = "epitech-zoidberg-dataset-validation"
    CONTAINER_TEST_NAME = "epitech-zoidberg-dataset-test"

    def __init__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(self.CONNECTION_STRING)
        self.container_train = self.blob_service_client.get_container_client(self.CONTAINER_TRAIN_NAME)
        self.container_val = self.blob_service_client.get_container_client(self.CONTAINER_VAL_NAME)
        self.container_test = self.blob_service_client.get_container_client(self.CONTAINER_TEST_NAME)

    def list_blobs(self, dataset: EDataset):
        if dataset == EDataset.TRAIN:
            return self.container_train.list_blobs()
        if dataset == EDataset.VALIDATION:
            return self.container_val.list_blobs()
        return self.container_test.list_blobs()

    def get_blob(self, dataset: EDataset, blob_name: str):
        if dataset == EDataset.TRAIN:
            return self.blob_service_client.get_blob_client(self.CONTAINER_TRAIN_NAME, blob_name)
        if dataset == EDataset.VALIDATION:
            return self.blob_service_client.get_blob_client(self.CONTAINER_VAL_NAME, blob_name)
        return self.blob_service_client.get_blob_client(self.CONTAINER_TEST_NAME, blob_name)
    
    def download_locally(self):
        local_download_path = f"{os.getcwd()}/dataset/"
        os.mkdir(os.path.join(local_download_path))
        container_train = [self.container_train, EDataset.TRAIN]
        container_val = [self.container_val, EDataset.VALIDATION]
        container_test = [self.container_test, EDataset.TEST]

        containers = [container_train, container_val, container_test]

        for container, dataset in containers:
            print(f"Downloading {dataset.name} dataset")
            os.mkdir(os.path.join(local_download_path, dataset.name))
            path = f"{os.path.join(local_download_path, dataset.name)}/"

            blobs = list(container.list_blobs())
            with tqdm(total=len(blobs), desc="Downloading", unit="file") as pbar:
                for blob in blobs:
                    download_file_path = os.path.join(path, blob.name)

                    with open(download_file_path, "wb") as download_file:
                        download_file.write(container.download_blob(blob.name).readall())
                    
                    pbar.update(1)