import csv
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import KFold

from local_types.enums import EDataset
from utils.data_loader import DataPreparer
from utils.metrics import Metrics
from models.resnet14 import ResNet14

class Trainer:
    def __init__(self, data_preparer:DataPreparer, batch_size=128, epochs=10, max_lr=1e-3, model:ResNet14=None, weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD, k_folds=5):
        self.history = []
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_lr = max_lr
        self.model = model
        self.data_preparer = data_preparer
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.opt_func = opt_func
        self.k_folds = k_folds


    def __get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    def fit(self, train_loader, val_loader):
        torch.cuda.empty_cache()

        optimizer = self.opt_func(self.model.parameters(), self.max_lr, weight_decay=self.weight_decay)

        # sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.max_lr, epochs=self.epochs, steps_per_epoch=len(train_loader))

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            lrs = []

            with tqdm(total=len(train_loader), desc="Training", unit="batch") as pbar:
                for batch in train_loader:
                    loss = ResNet14.training_step(self.model, batch)
                    train_losses.append(loss)
                    loss.backward()
                    if self.grad_clip:
                        nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)

                    optimizer.step()
                    optimizer.zero_grad()

                    # lrs.append(self.__get_lr(optimizer))
                    # sched.step()
                    pbar.update(1)

            # Validation phase
            result = ResNet14.evaluate(self.model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            # result['lr'] = lrs
            ResNet14.epoch_end(self.model, epoch, result)
            self.history.append(result)

    def cross_validation(self):
        kfold = KFold(n_splits=self.k_folds, shuffle=True)

        for fold, (train_ids, val_ids) in enumerate(kfold.split(self.data_preparer.fold_ds)):
            print(f'====== FOLD {fold} ======')

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

            train_dl = self.data_preparer.get_dataloaders(dataset_type=EDataset.FOLD, batch_size=self.batch_size, sampler=train_subsampler)
            val_dl = self.data_preparer.get_dataloaders(dataset_type=EDataset.FOLD, batch_size=self.batch_size, sampler=val_subsampler)

            self.fit(train_loader=train_dl, val_loader=val_dl)

    def save_history_to_csv(self, file_name):
        file_meta_data = [("val_loss", "val_acc", "train_loss", "f1", "roc_auc")]
        file_data_csv = [(i["val_loss"], i["val_acc"], i["train_loss"], i["f1"], i["roc_auc"]) for i in self.history]
        full_file = file_meta_data + file_data_csv
        path = os.path.join(os.getcwd(), f"src/processed/{file_name}.csv")

        with open(path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(full_file)
                    