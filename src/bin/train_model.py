import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

from utils.data_loader import DataPreparer

from utils.data_loader import DeviceDataLoader
from utils.data_loader import DataPreparer
from models.resnet14 import ResNet14
from utils.trainer import Trainer

from local_types.enums import EDataset

batch_size = 200
data_preparer = DataPreparer(image_size=75)
data_preparer.load_datasets(locally=True)

device = DeviceDataLoader.get_default_device()
model = DeviceDataLoader.to_device(ResNet14(3,3), device)
model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))


trainer = Trainer(
    model=model,
    data_preparer=data_preparer,
    batch_size=batch_size,
    max_lr=1e-2,
    epochs=2,
    k_folds=10,
    grad_clip=1,
    opt_func=torch.optim.SGD,
)

trainer.cross_validation()
# trainer.fit(
#     data_preparer.get_dataloaders(dataset_type=EDataset.TRAIN, batch_size=batch_size), 
#     data_preparer.get_dataloaders(dataset_type=EDataset.VALIDATION, batch_size=batch_size)
# )

ResNet14.save_weights_and_biases(model, file_name="resnet14_general_best_cross_val_3")
# ResNet14.save_weights_and_biases(model, file_name="resnet14_weights", path=os.path.join(os.getcwd(), "cloud", "deployment"))
trainer.save_history_to_csv(file_name="resnet14_general_best_cross_val_3")
