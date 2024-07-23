import torch
from sklearn.metrics import f1_score, roc_auc_score

class Metrics:
    def __init__(self, outputs):
        batch_preds = [x['preds'] for x in outputs]
        batch_probs = [x['probs'] for x in outputs]
        batch_labels = [x['labels'] for x in outputs]

        self.preds = torch.cat(batch_preds)
        self.labels = torch.cat(batch_labels)
        self.probs = torch.cat(batch_probs)

    @staticmethod
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def f1_score(self):
        return f1_score(self.labels.cpu(), self.preds.cpu(), average='macro')

    def roc_auc_score(self):
        return roc_auc_score(self.labels.cpu().numpy(), self.probs.cpu().numpy(), multi_class='ovo')
