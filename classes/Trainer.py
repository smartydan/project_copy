from datetime import datetime
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import matplotlib.pyplot as plt

from itertools import product
from IPython.display import clear_output

import numpy as np

import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader

from .MyDataset import MyDataset
from .MyModel import MyModel


class Trainer:
    """Class for model training and validation."""

    def __init__(self, model, device, train, validate, test, params=None, dataset=MyDataset, dataloader=DataLoader,
                 preprocessor=None,
                 lossf=nn.CrossEntropyLoss,
                 optimizer=torch.optim.Adam, batch_size=10,
                 base_dir='data/',
                 best_model_path='best_model',
                 train_window_size=10,
                 test_window_size=5,
                 max_len=512,
                 epochs=10,
                 learning_rate=0.0005, metric_to_use='f1', min_spoil=1, timer=1, scheduler=None):
        """
        :param model: model to train
        :param device: device used for model training
        :param train: train data
        :param validate: validation data
        :param test: test data
        :param params: grid of hyperparams for best model selection
        :param dataset: dataset derived from torch.utils.data.Dataset
        :param dataloader: dataloader derived from torch.utils.data.DataLoader
        :param preprocessor: preprocessor to pass to dataset 
        :param lossf: loss function to use
        :param optimizer: optimizer to use
        :param base_dir: base directory to save data to
        :param best_model_path: best model will be saved at f'{base_dir}{best_model_tpath}'
        :param train_window_size: number of train batches for average score calculation
        :param train_window_size: number of test batches for average score calculation
        :param max_len: maximal length for Tokenizer
        :param epochs: number of epochs for model training
        :param learning_rate: learning rate
        :param metric_to_use: which metric should be considered for best model choosing
        :param min_spoil: lower bound for max_spoil
        :param timer: each 'timer' epochs max_spoil is lowered
        :param scheduler: scheduler to use (None by default)
        """

        try:
            self.model = model(device, max_len=max_len).to(device)
        except OSError:
            print("Could not load model to device")

        self.train_dataset = dataset(train.copy(), preprocessor=preprocessor)
        self.test_dataset = dataset(test.copy(), preprocessor=preprocessor)
        self.dataloader = dataloader

        self.max_len = max_len

        try:
            self.train_loader = dataloader(self.train_dataset, batch_size=batch_size)
            self.test_loader = dataloader(self.test_dataset, batch_size=batch_size)
        except OSError:
            print("Could not load dataset")

        if metric_to_use not in ['f1', 'accuracy', 'precision', 'recall']:
            raise ValueError("metric_to use must be one of 'f1', 'accuracy', 'precision', 'recall'")

        self.train_ = train
        self.test_ = test
        self.validate = validate

        self.device = device
        self.lr = learning_rate
        self.batch_size = batch_size
        self.base_dir = base_dir
        self.best_model_path = best_model_path
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.min_spoil = min_spoil
        self.timer = timer

        self.lossf = lossf()
        self.opt = optimizer
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.metric_to_use = metric_to_use
        self.best_metric = -float('inf')

        self.epochs = epochs
        self.params = params
        self.num = 0

        self.data = dict()

        self.scheduler = None if scheduler is None else scheduler()

        self.trial_run()

    def trial_run(self):
        inputs = "0" * self.max_len
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.lossf(outputs, outputs)
        loss.backward()

    def reinitialize(self, batch_size, lr, max_spoil, spoil_proba):
        self.model.reinitialize()
        self.model.to(self.device)

        self.train_loader = self.dataloader(self.train_dataset, batch_size=batch_size)
        self.test_loader = self.dataloader(self.test_dataset, batch_size=batch_size)

        for loader in self.train_loader, self.test_loader:
            loader.dataset.max_spoil = max_spoil
            loader.dataset.spoil_proba = spoil_proba

        self.optimizer = self.opt(self.model.parameters(), lr=lr)

    def choose_model(self):
        for val in product(*self.params.values()):
            params = {k: v for k, v in zip(self.params, val)}
            self.reinitialize(**params)
            self.data[self.num] = dict()
            self.data[self.num]['params'] = params
            self.train(self.epochs)
            self.num += 1

    def check(self, epoch):
        if epoch % self.timer == 0:
            if self.test_loader.dataset.max_spoil > self.min_spoil:
                self.test_loader.dataset.max_spoil -= 1

    def train_one_epoch(self, epoch, verbose=False):
        self.check(epoch)

        cur_loss = sum_loss = 0

        all_outputs = torch.empty(0).to('cpu')
        all_labels = torch.empty(0).to('cpu')

        for i, data in enumerate(tqdm(self.train_loader), 1):
            inputs, labels = data
            labels = labels.long().to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.lossf(outputs, labels)

            if self.scheduler:
                self.scheduler.step()

            loss.backward()
            self.optimizer.step()
            loss_ = loss.item()

            if self.num not in self.data:
                self.data[self.num] = dict()

            if 'train_loss' not in self.data[self.num]:
                self.data[self.num]['train_loss'] = []

            self.data[self.num]['train_loss'].append(loss_)
            cur_loss += loss_

            all_outputs = torch.cat((all_outputs, outputs.to('cpu')), dim=0)
            all_labels = torch.cat((all_labels, labels.to('cpu')), dim=0)

            if i % self.train_window_size == 0:
                last_loss = cur_loss / self.train_window_size
                sum_loss += last_loss
                cur_loss = 0
                if verbose:
                    print(f"batch {i}\nloss: {last_loss:.5}")

        metric = self.calculate_scores(all_labels, all_outputs, verbose=verbose, on_train=True)

        return sum_loss / len(self.train_loader), metric

    def validate_one_epoch(self, epoch, verbose=False):
        sum_loss = 0

        all_outputs = torch.empty(0).to('cpu')
        all_labels = torch.empty(0).to('cpu')

        with torch.no_grad():
            for i, vdata in enumerate(tqdm(self.test_loader), 1):
                inputs, labels = vdata

                outputs = self.model(inputs)
                labels = labels.long().to(self.device)

                all_outputs = torch.cat((all_outputs, outputs.to('cpu')), dim=0)
                all_labels = torch.cat((all_labels, labels.to('cpu')), dim=0)

                loss_ = self.lossf(outputs, labels).item()

                if self.num not in self.data:
                    self.data[self.num] = dict()

                if 'test_loss' not in self.data[self.num]:
                    self.data[self.num]['test_loss'] = []

                self.data[self.num]['test_loss'].append(loss_)
                sum_loss += loss_

                if verbose and i % self.test_window_size == 0:
                    print(f"Batch {i} validation")
                    self.calculate_scores(labels, outputs, on_train=False, verbose=verbose)

        if verbose:
            print(f"Calculating epoch {epoch + 1} validation scores")

        metric = self.calculate_scores(all_labels, all_outputs, on_train=False, verbose=verbose)

        return sum_loss / len(self.train_loader), metric

    def train(self, epochs, verbose=False, plot=True, save=False):
        for epoch in range(epochs):
            time = datetime.now()

            if verbose:
                print(f"Epoch {epoch + 1} running")

            self.model.train(True)
            loss, metric = self.train_one_epoch(epoch, verbose=verbose)

            if plot:
                clear_output(wait=True)
                self.plot_loss(on_train=True)
                self.plot_metrics(on_train=True)

            self.model.eval()
            v_loss, v_metric = self.validate_one_epoch(epoch, verbose=verbose)

            if plot:
                clear_output(wait=True)
                self.plot_loss(on_train=False)
                self.plot_metrics(on_train=False)

            if verbose:
                print(f'Time taken for epoch {epoch + 1}: {datetime.now() - time}\n')

            if save:
                self.save_model(v_metric, epoch, verbose)

    def save_model(self, metric, epoch, verbose):
        if metric > self.best_metric:
            self.best_metric = metric
            torch.save(self.model.state_dict(), f'{self.base_dir}{self.best_model_path}')
            if verbose:
                print(f'Saving best model with {self.metric_to_use} {metric}')

        path = f'{self.base_dir}/epoch_{epoch}_num_{self.num}'
        torch.save(self.model.state_dict(), path)

    def load_model(self, model_path):
        model = MyModel(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def plot_loss(self, on_train=False):
        if on_train:

            data = self.data[self.num]['train_loss']

            plt.plot(data)
            plt.title('Training loss')

            step = len(self.train_loader)
            for el in range(step, len(data), step):
                plt.axvline(x=el, color='red', linestyle='--')
        else:

            data = self.data[self.num]['test_loss']

            plt.plot(data)
            plt.title('Validation loss')

            step = len(self.test_loader)
            for el in range(step, len(data), step):
                plt.axvline(x=el, color='red', linestyle='--')

        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.show()

    def plot_metrics(self, on_train=False):
        it = self.data[self.num]['train_metrics'] if on_train else self.data[self.num]['test_metrics']
        for metric, values in it.items():
            plt.plot(values, label=metric)
            plt.xlabel('Epochs')
        plt.legend(loc='best')
        plt.show()

    def calculate_scores(self, y_true, y_pred, on_train=False, verbose=False):
        y_true = y_true.cpu().detach().numpy()
        y_pred = np.argmax(y_pred.cpu().detach().numpy(), axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if verbose:
            print(f'accuracy: {accuracy:.5f} precision: {precision:.5f} '
                  f'recall: {recall:.5f} f1 score: {f1:.5f}')

        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

        if not self.num in self.data:
            self.data[self.num] = dict()
        if on_train:
            if not 'train_metrics' in self.data[self.num]:
                self.data[self.num]['train_metrics'] = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

            for metric_name, metric in metrics.items():
                self.data[self.num]['train_metrics'][metric_name].append(metric)
        else:
            if not 'test_metrics' in self.data[self.num]:
                self.data[self.num]['test_metrics'] = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

            for metric_name, metric in metrics.items():
                self.data[self.num]['test_metrics'][metric_name].append(metric)

        return metrics[self.metric_to_use]

    def save(self):
        with open(f'{self.base_dir}/data.pkl', 'wb') as f:
            pickle.dump(self.data, f)
