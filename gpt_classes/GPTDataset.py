from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):

    def __init__(self, df, preprocessor, args, id_column='document.id', topic=False, num=None, fixed=True):
        """
        :param df: input data in DataFrame format
        :param preprocessor: class to generate prompt and decription
        :param id_column: column to set index on (default is None)
        :param args: args to consider
        :param topic: if True topics will be included in the decription
        :param num: number of args to include in prompt, if None, all will be included
        :param fixed: if False, args will be choosen on each iteration
        """

        self.df = df.copy()

        if id_column:
            self.id_col = id_column
            self.df.set_index(id_column, inplace=True)

        self.topic = topic
        self.preprocessor = preprocessor
        self.unique_ids = self.df.index.unique()
        self.args=args
        self.num = min(num or 0, len(args))
        self.fixed = fixed
        self.fixed_args = None

        if num and num > 0 and fixed:
            self.fixed_args = np.random.choise(self.args, size=num, replace=False)
        
    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.unique_ids)

    def __getitem__(self, idx):
        """
        :param idx: index of the sample to be returned
        :return: prompt
        """
        id_ = self.unique_ids[idx]

        if self.num > 0 and not self.fixed:
            self.fixed_args = np.random.choise(self.args, size=self.num, replace=False)

        prompt, description, text = self.preprocessor.fit(id_=id_, topic=self.topic, topic_spoil=0, spoil_size=0, prompt=True, prompt_list=self.fixed_args)

        return f'Задание: {prompt or ""}\nТекст: {text or ""}\nОписание: {description or ""}',  f'Задание: {prompt or ""}\nТекст: {text or ""}\nОписание:'