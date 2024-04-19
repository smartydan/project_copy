from torch.utils.data import Dataset

import numpy as np


class MyDataset(Dataset):
    """Custom class to store data as a PyTorch dataset"""

    def __init__(self, df, preprocessor, id_column='document.id', max_spoil=float('inf'), topic_spoil=0,
                 tokenizer=None, topic=False, spoil_proba=0.5):
        """
        :param df: input data in DataFrame format
        :param id_column: column to set index on (default is None)
        :param max_spoil: maximum of features to spoil
        :param topic_spoil: maximum number of topics to add if spoiled
        :param tokenizer: tokenizer to use for word tokenization
        :param topic: if True topics will be included in a decription
        :param spoil_proba: probability of spoling a description
        """

        self.df = df.copy()
        self.preprocessor = preprocessor

        if id_column:
            self.id_col = id_column
            self.df.set_index(id_column, inplace=True)
    
        self.max_spoil = min(max_spoil, len(self.preprocessor.args))
        self.topic_spoil = topic_spoil
        self.tokenizer = tokenizer
        self.topic = topic
        self.spoil_proba = spoil_proba
        self.unique_ids = self.df.index.unique()
        
    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.unique_ids)

    def __getitem__(self, idx):
        """
        :param idx: index of the sample to be returned
        :return: [description, target]
        """
        id_ = self.unique_ids[idx]
        spoil_size = topic_spoil = 0

        if np.random.random() < self.spoil_proba:
            spoil_size = self.max_spoil
            topic_spoil = self.topic_spoil

        description, text = self.preprocessor.fit(id_=id_, topic=self.topic, topic_spoil=topic_spoil,
                                                  spoil_size=spoil_size)
        return f'Текст: {text or ""} [SEP] Описание: {description or ""}', spoil_size + topic_spoil > 0
