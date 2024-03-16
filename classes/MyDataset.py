from torch.utils.data import Dataset

import numpy as np


class MyDataset(Dataset):
    """Custom class to store data as a PyTorch dataset"""

    def __init__(self, df, preprocessor, id_column='document.id', max_spoil=float('inf'), topic_spoil=0,
                 tokenizer=None, topic=False, spoil_proba=0.5):
        """
        :param df: input data in DataFrame format
        :param tokenizer: tokenizer to use for word tokenization
        :param spoil_proba: probability of spoiling a description
        """
        self.df = df
        self.tokenizer = tokenizer

        if id_column is not None:
            self.id_col = id_column
            self.df.set_index(id_column, inplace=True)

        self.spoil_proba = spoil_proba
        self.preprocessor = preprocessor
        self.max_spoil = min(max_spoil, len(self.preprocessor.args))
        self.topic_spoil = topic_spoil
        self.topic = topic

    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.df.index)

    def __getitem__(self, idx):
        """
        :param idx: index of the sample to be returned
        :return: [sample, target]
        """
        id_ = self.df.index[idx]

        spoil_size = 0
        topic_spoil = 0
        if np.random.random() < self.spoil_proba:
            spoil_size = self.max_spoil
            topic_spoil = self.topic_spoil

        description, text = self.preprocessor.fit(id_=id_, topic=self.topic, topic_spoil=topic_spoil,
                                                  spoil_size=spoil_size)
        return 'Текст: ' + (text or '') + '[SEP]' + ' Описание: ' + (description or ''), (
                spoil_size + topic_spoil > 0)
