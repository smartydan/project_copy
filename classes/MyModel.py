from torch import nn
from transformers import AutoTokenizer, AutoModel


class MyModel(nn.Module):
    """Pretrained model with custom text classificator."""

    def __init__(self, device, max_len=512, pretrained_model_path="cointegrated/rubert-tiny2"):
        """
        :param device: device to use
        :param max_len: maximal length for Tokenizer
        :param pretrained_model_path: model to load for fine-tuning
        """
        super().__init__()
        try:
            self.model = AutoModel.from_pretrained(pretrained_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, model_max_length=max_len)
        except OSError:
            print("Could not load pretrained model")
        self.classifier = nn.Sequential(nn.Linear(self.model.config.hidden_size, 2))
        self.device = device
        self.max_len = max_len
        self.path = pretrained_model_path

    def reinitialize(self):
        """
        Reloads model and tokenizer
        """
        try:
            self.model = AutoModel.from_pretrained(self.path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.path, model_max_length=self.max_len)
        except OSError:
            print("Could not load pretrained model")
        self.classifier = nn.Sequential(nn.Linear(self.model.config.hidden_size, 2))

    def forward(self, x):
        """
        :param x: input text
        :return: tensor of output data
        """
        tokenized = self.tokenizer(x, padding=True, truncation=True, return_tensors="pt").to(self.device)  # токенизируем
        output = self.model(**tokenized)  # вызываем модель
        return self.classifier(output.last_hidden_state[:, 0, :])  # вызываем классификатор
