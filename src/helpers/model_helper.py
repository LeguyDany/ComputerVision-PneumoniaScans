import os
import tarfile
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import Metrics


class ImageClassificationBase(nn.Module):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = Metrics.accuracy(out, labels)
        _, preds = torch.max(out, dim=1)
        probs = F.softmax(out, dim=1)
        return {'val_loss': loss.detach(), 'val_acc': acc, 'preds': preds, 'labels': labels, 'probs': probs}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()

        metrics = Metrics(outputs)

        f1 = metrics.f1_score()
        roc_auc = metrics.roc_auc_score()

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'f1': f1, 'roc_auc': roc_auc}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, f1_score:{:.4f}, roc_auc_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc'], result['f1'], result['roc_auc']))

    def save_weights_and_biases(self, file_name:str, path:str=None):
        if path != None:
            model_path= os.path.join(path, f"{file_name}.pth")
        else: 
            model_path= os.path.join(os.getcwd(), f"src/processed/{file_name}.pth")

        state_dict = {k.replace("module.", ""): v for k, v in self.state_dict().items()}
        torch.save(state_dict, model_path)

    def generate_tarball(self, initial_folder_path:str, output_path:str):
        model_file_serve = os.path.join(
            f"{os.path.basename(initial_folder_path)}.tar.gz",
            output_path
        )

        with tarfile.open(model_file_serve, 'w:gz') as tar:
            tar.add(initial_folder_path, arcname=os.path.basename(initial_folder_path))

    def load(self, file_name, path = None):
        if path:
            model_path = os.path.join(path, f"{file_name}.pth")
        else:
            model_path = os.path.join(os.getcwd(), f"src/processed/{file_name}")

        self.load_state_dict(torch.load(model_path))

    @staticmethod
    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [ImageClassificationBase.validation_step(model, batch) for batch in val_loader]
        return ImageClassificationBase.validation_epoch_end(model, outputs)
