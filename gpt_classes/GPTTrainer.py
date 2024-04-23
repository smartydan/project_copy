from datetime import datetime
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchtext.data.metrics import bleu_score 

from itertools import product
from IPython.display import clear_output

import numpy as np

import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader


class GPTTrainer:
    """Class for model training and validation."""

    def __init__(self, model, device, train, validate, test, dataset, args, dataloader,
                 save=True,
                 preprocessor=None,
                 optimizer=torch.optim.Adam, batch_size=10,
                 base_dir='data/',
                 best_model_path='best_model',
                 train_window_size=10,
                 test_window_size=5,
                 max_len=512,
                 epochs=10, number_of_generations=10,
                 learning_rate=0.0005, metric_to_use='bleu_score', min_spoil=1, timer=1):
        """
        :param model: model to train
        :param device: device used for model training
        :param train: train data
        :param validate: validation data
        :param test: test data
        :param dataset: dataset derived from torch.utils.data.Dataset
        :param args: args for dataset (structure specifical) 
        :param dataloader: dataloader derived from torch.utils.data.DataLoader
        :param save: if True, model will be saved each epoch
        :param preprocessor: preprocessor to pass to dataset 
        :param optimizer: optimizer to use
        :param base_dir: base directory to save data to
        :param best_model_path: best model will be saved at f'{base_dir}{best_model_tpath}'
        :param train_window_size: number of train batches for average score calculation
        :param train_window_size: number of test batches for average score calculation
        :param max_len: maximal length for Tokenizer
        :param epochs: number of epochs for model training
        :param learning_rate: learning rate
        :param metric_to_use: which metric should be considered for best model choosing
        """

        try:
            self.model = model(device, max_len=max_len).to(device)
        except OSError:
            print("Could not load model to device")

        self.preprocessor = preprocessor
        self.train_dataset = dataset(train.copy(), args=args, preprocessor=preprocessor)
        self.test_dataset = dataset(test.copy(), args=args, preprocessor=preprocessor)
        self.dataloader = dataloader
        self.args = args

        self.max_len = max_len

        try:
            self.train_loader = dataloader(self.train_dataset, batch_size=batch_size)
            self.test_loader = dataloader(self.test_dataset, batch_size=batch_size)
        except OSError:
            print("Could not load dataset")

        if metric_to_use not in ['bleu_score']:
            raise ValueError("metric_to use must be one of 'bleu_score'")

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

        self.save = save

        self.opt = optimizer
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.metric_to_use = metric_to_use
        self.best_metric = -float('inf')

        self.epochs = epochs
        self.num = 0

        self.number_of_gen = number_of_generations
        self.ids = np.random.choice(range(len(self.test_loader)), size=min(number_of_generations, len(self.test_loader)), replace=False)

        self.data = dict()


    def reinitialize(self, batch_size, lr, ngrams):
        self.model.reinitialize(ngrams)
        self.model.to(self.device)

        self.train_loader = self.dataloader(self.train_dataset, batch_size=batch_size)
        self.test_loader = self.dataloader(self.test_dataset, batch_size=batch_size)

        self.optimizer = self.opt(self.model.parameters(), lr=lr)

    def choose_model(self, params):
        for val in product(*params.values()):
            params_ = {k: v for k, v in zip(params, val)}
            self.reinitialize(**params_)
            self.data[self.num] = dict()
            self.data[self.num]['params'] = params_
            self.train(self.epochs)
            self.num += 1

    def train_one_epoch(self, epoch, verbose=False):

        cur_loss = sum_loss = 0

        for i, (data, cut_data) in enumerate(tqdm(self.train_loader), 1):
            self.optimizer.zero_grad()
            outputs = self.model(data, cut_data)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            loss_ = loss.item()

            if self.num not in self.data:
                self.data[self.num] = dict()

            if 'train_loss' not in self.data[self.num]:
                self.data[self.num]['train_loss'] = []

            self.data[self.num]['train_loss'].append(loss_)
            cur_loss += loss_

            if i % self.train_window_size == 0:
                last_loss = cur_loss / self.train_window_size
                sum_loss += last_loss
                cur_loss = 0
                if verbose:
                    print(f"batch {i}\nloss: {last_loss:.5}")
                    
        return sum_loss / len(self.train_loader)

    def validate_one_epoch(self, epoch, verbose=False):
        sum_loss = 0

        with torch.no_grad():
            for i, (data, cut_data) in enumerate(tqdm(self.test_loader), 1):
                outputs = self.model(data, cut_data)
                loss_ = outputs.loss.item()

                if self.num not in self.data:
                    self.data[self.num] = dict()

                if 'test_loss' not in self.data[self.num]:
                    self.data[self.num]['test_loss'] = []

                self.data[self.num]['test_loss'].append(loss_)
                sum_loss += loss_

                if verbose and i % self.test_window_size == 0:
                    print(f"Batch {i} validation")

        for id_ in self.ids:
            if epoch not in self.data[self.num]:
                self.data[self.num][epoch] = {}
            if 'generated' not in self.data[self.num][epoch]:
                self.data[self.num][epoch]['generated'] = []
            
            self.data[self.num][epoch]['generated'].append((id_, self.model.generate(self.test_dataset[id_]))) # добавляем id и сгенерированное описание

        if verbose:
            print(f"Calculating epoch {epoch + 1} validation scores")

        return sum_loss / len(self.train_loader)

    def train(self, epochs, verbose=False, plot=True, save=False):
        for epoch in range(epochs):

            if verbose:
                time = datetime.now()
                print(f"Epoch {epoch + 1} running")

            self.model.train(True)
            loss = self.train_one_epoch(epoch, verbose=verbose)

            if plot:
                clear_output(wait=True)
                self.plot_loss(on_train=True)
            
            self.model.eval()
            v_loss = self.validate_one_epoch(epoch, verbose=verbose)

            if plot:
                clear_output(wait=True)
                self.plot_loss(on_train=False)

            if verbose:
                print(f'Time taken for epoch {epoch + 1}: {datetime.now() - time}\n')

            if save or self.save:
                self.save_model(v_loss, epoch, verbose)

    def save_model(self, metric, epoch, verbose):
        path = f'{self.base_dir}/epoch_{epoch}_num_{self.num}_loss_{metric}'
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

    def save(self):
        with open(f'{self.base_dir}/data.pkl', 'wb') as f:
            pickle.dump(self.data, f)