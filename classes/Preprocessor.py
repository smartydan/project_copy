import numpy as np


class Preprocessor:
    """Class to aggregate features and give a description of a certain spoil degree"""

    def __init__(self, df, args, var_vocab, topic_to_russian, people_num=1):
        """
        :param df: input data in DataFrame format
        :param args: arguments that can be aggregated
        :param var_vocab: description of the given value of the given argument
        :param topic_to_russian: topic to russian translation
        :param people_num: number of assessors needed to consider an aggregated parameter as valid
        """
        self.df = df.drop_duplicates(subset=['document.id', 'assessor', 'seed_eth_group'])
        self.args = args
        self.var_vocab = var_vocab
        self.topic_to_russian = topic_to_russian
        self.people_num = people_num

        self.description = ''
        self.data = None

    def define(self, x):
        """
        :return: the mode if it's unique and considered as true value
            by least people_num people else None
        """
        counts = x.value_counts(dropna=False)
        mode = counts.iloc[0]
        if mode >= self.people_num and np.sum(counts == mode) == 1:
            return counts.index[0]
        return None

    def add_info(self, var, spoil, ethnicity=None):
        """
        adds information about variable to the description
        :param var: variable to get info about
        :param spoil: if True then value will be spoiled
        :param ethnicity: passes, if the variable is ethnicity-leveled
        """

        if ethnicity:
            data = self.data[self.data['seed_eth_group'] == ethnicity][var]
        else:
            data = self.data.drop_duplicates(subset='assessor')[var]
        value = self.define(data)
        labels = self.var_vocab[var]['labels']
        if value and value in labels:
            if spoil:
                value = np.random.choice(list(set(labels) - {value}))
            desc = labels[value]
            if desc:
                if self.description:
                    self.description += ', '

                if ethnicity:
                    self.description += desc.format(ethnicity)
                else:
                    self.description += desc

    def get_topics(self, topic_spoil):
        """
        adds information about topics
        :param topic_spoil: probability of spoiling a topic
        """
        add_topics = ''
        topics_data = self.data.filter(regex="has_topic*")
        topics = topics_data.columns
        true_topics = []
        false_topics = []

        for topic in topics:
            value = self.define(topics_data[topic])
            if value == 1:
                true_topics.append(topic)
            elif value is not None:
                false_topics.append(topic)

        for topic in true_topics:
            if false_topics and np.random.random() < topic_spoil:
                topic = false_topics.pop(0)

            if add_topics:
                add_topics += ', '
            add_topics += self.topic_to_russian[topic[10:]]

        if add_topics:
            self.description += 'Текст имеет темы: ' + add_topics

    def get_prompt(self, var, eths):
        if self.var_vocab[var]['aspect_level']:
            return ", ".join(self.var_vocab[var]['prompt'].format(eth) for eth in eths)
        return self.var_vocab[var]['prompt']

    def fit(self, id_, not_to_spoil=None, spoil_size=0, topic=False, list_ethnicities=False, topic_spoil=0, prompt=False, prompt_list=None):
        """
        :param id_: id of a sample to describe
        :param not_to_spoil: parametes that won't be spoil anyway
        :param spoil_size: numer of features to spoil
        :param topic: if True, topics will be included into description
        :param list_ethnicities: if True, list of mentioned ethincities will be included into description
        :param topic_spoil: spoil percent for topics
        :param prompt: if True, prompt will be generated at the beginning of return
        :param prompt_list: list of args to include in prompt, if None, all will be included
        :return: description of a given text
        """

        self.data = self.df.loc[self.df['document.id'] == id_]
        sz = self.data.shape[0]

        if sz == 0:
            print("No such id")
            return None, None

        not_to_spoil = not_to_spoil or []

        self.description = ''
        self.prompt = ''

        eths = self.data['seed_eth_group'].unique()

        if prompt:
            self.prompt += 'Сгенерируй описание следующего текста, оцени'
            if prompt_list:
                self.prompt += ", ".join(self.get_prompt(arg, eths) for arg in prompt_list if arg in self.args)
            else:
                self.prompt += ", ".join(self.get_prompt(arg, eths) for arg in self.args)
            self.prompt += "."

        to_spoil = np.random.choice(list(set(self.args) - set(not_to_spoil)), size=spoil_size, replace=False)

        if list_ethnicities:
            self.description += f'В этом тексте упоминается {", ".join(eth for eth in eths)}.'

        if topic:
            self.get_topics(topic_spoil)

        for var in self.args:
            if self.var_vocab[var]['aspect_level']:
                for eth in eths:
                    self.add_info(var, var in to_spoil, eth)
            else:
                self.add_info(var, var in to_spoil)

        self.description += '\n'

        return (self.prompt, self.description, self.data.iloc[0].source_text) if prompt else (self.description, self.data.iloc[0].source_text)