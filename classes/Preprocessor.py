import numpy as np


class Preprocessor:

    def __init__(self, df, args, var_vocab, topic_to_russian, people_num=1):
        self.df = df
        self.args = args
        self.var_vocab = var_vocab
        self.topic_to_russian = topic_to_russian
        self.people_num = people_num
        self.description = ''
        self.data = None

    def define(self, x):
        counts = x.value_counts(dropna=False)
        mode = counts.iloc[0]
        if np.sum(counts == mode) == 1 and mode >= self.people_num:
            return counts.index[0]
        return None  # to be explicit

    def add_info(self, var, spoil, ethnicity=None):
        if ethnicity:
            data = self.data[self.data['seed_eth_group'] == ethnicity][var]
        else:
            data = self.data[var]
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
        add_topics = ''
        topics_data = self.data.filter(regex="has_topic*")
        topics = topics_data.columns
        true_topics = []
        false_topics = []

        for topic in topics:
            value = self.define(topics_data[topic])
            if value:
                true_topics.append(topic)
            elif value is not None:
                false_topics.append(topic)

        # corrected_proba = (1 - topic_spoil) ** (1 / len(true_topics))  # something is happening here
        for topic in true_topics:
            if false_topics and np.random.random() < topic_spoil:
                topic = false_topics.pop(0)

            if add_topics:
                add_topics += ', '
            add_topics += self.topic_to_russian[topic[10:]]

        if add_topics:
            self.description += 'Текст имеет темы: ' + add_topics

    def fit(self, id_, not_to_spoil=None, spoil_size=0, topic=False, list_ethnicities=False, topic_spoil=0):

        self.data = self.df.loc[self.df['document.id'] == id_].drop_duplicates(subset='assessor')
        sz = self.data.shape[0]

        if sz == 0:
            print("No such id")
            return

        not_to_spoil = not_to_spoil or []

        self.description = ''

        to_spoil = np.random.choice(list(set(self.args) - set(not_to_spoil)), size=spoil_size, replace=False)
        eths = self.data['seed_eth_group'].unique()

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

        return self.description, self.data.iloc[0].source_text
