import os
import numpy as np
from matplotlib import pyplot as plt
import torch

from local_types.enums import EDataset, EOutput
from utils.data_loader import DataPreparer
from utils.data_loader import DeviceDataLoader

import plotly.express as px
import pandas as pd


class Visualizer:
    
    @staticmethod
    def visualize_random(dataset_type: EDataset, data_preparer: DataPreparer):

        if(dataset_type == EDataset.TRAIN):
            data = data_preparer.train_ds
        elif(dataset_type == EDataset.VALIDATION):
            data = data_preparer.validation_ds
        elif(dataset_type == EDataset.TEST):
            data = data_preparer.test_ds
        else:
            data = data_preparer.fold_ds

        image, label = data[np.random.randint(0, len(data))]
        Visualizer.visualize(image, label)

    @staticmethod
    def visualize_tensor(image, label):
        plt.imshow(image.permute(0, 2, 3, 1))
        plt.title(EOutput(label).name)
        plt.axis('off')
        plt.show()

    @staticmethod
    def predict(model, image):
        image_to_tensor = (torch.from_numpy(image).float()) / 255.0
        image_to_tensor = image_to_tensor.unsqueeze(0)
        image_to_tensor = image_to_tensor.permute(0, 3, 1, 2).to(DeviceDataLoader.get_default_device())
        return model(image_to_tensor)

    @staticmethod
    def visualize(image, label):
        plt.imshow(image)
        plt.title(EOutput(label).name)
        plt.axis('off')
        plt.show()

    @staticmethod
    def visualize_predictions(model, image, label):
        pred = Visualizer.predict(model, image)
        print(f"Label: {EOutput(label).name}, prediction: {EOutput(torch.max(pred, dim=1)[1].item()).name}")

        Visualizer.visualize(image, label)

    @staticmethod
    def visualize_history_scores(history):
        val_acc = [x['val_acc'] for x in history]
        val_f1 = [x['f1'] for x in history]
        val_roc = [x['roc_auc'] for x in history]
        plt.plot(val_acc, '-bx')
        plt.plot(val_f1, '-rx')
        plt.plot(val_roc, '-gx')
        plt.xlabel('epoch')
        plt.ylabel('metrics')
        plt.legend(['Accuracy', 'F1 Score', 'ROC AUC'])
        plt.title('Metrics vs. No. of epochs')

    @staticmethod
    def visualize_several_csv(file_names, metric, column_names, title, x_name, y_name,path=None):
        """ metrics: val_acc, f1, roc_auc, val_loss, train_loss """

        columns = ["val_acc", "f1", "roc_auc", "val_loss", "train_loss"]
        columns.remove(metric)

        dfs = pd.DataFrame()

        for file_name in file_names:
            if path:
                csv_path = os.path.join(path, file_name)
            else:
                csv_path = os.path.join(os.getcwd(), "src/processed", file_name)

            history_df = pd.read_csv(csv_path)
            # history_df = history_df.head(2)
            df = history_df.drop(columns=columns)
            df = df.reset_index()

            column_name = column_names[file_names.index(file_name)]
            dfs[column_name] = history_df[metric]

        fig = px.line(dfs, x=dfs.index, y=dfs.columns, title=title)
        fig.update_xaxes(title_text=x_name)
        # fig.update_yaxes(title_text=y_name, range=[0, 2.5])
        # fig.update_yaxes(title_text=y_name)
        fig.show()

    @staticmethod
    def visualize_history_from_csv(file_name, have_score=True, have_loss=True, path=None):
        if path:
            csv_path = os.path.join(path, file_name)
        else:
            csv_path = os.path.join(os.getcwd(), "../processed", file_name)

        history_df = pd.read_csv(csv_path)

        scores_df = history_df.drop(columns=["val_loss", "train_loss"])
        loss_df = history_df.drop(columns=["val_acc", "f1", "roc_auc"])

        if(have_score and have_loss):
            df_melt = history_df.reset_index().melt(id_vars='index', var_name='Variables', value_name='Value')
        elif(have_loss):
            df_melt = loss_df.reset_index().melt(id_vars='index', var_name='Variables', value_name='Value')
        else:
            df_melt = scores_df.reset_index().melt(id_vars='index', var_name='Variables', value_name='Value')

        fig = px.line(df_melt, x='index', y='Value', color='Variables', title='Evolution of metrics over time')
        fig.update_xaxes(title_text="Epochs")
        fig.show()