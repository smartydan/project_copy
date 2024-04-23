from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class GPTModel(nn.Module):
    """Pretrained model with custom text classificator."""

    def __init__(self, device, max_len=512, max_len_model=100, ngram=2, pretrained_model_path="ai-forever/rugpt3small_based_on_gpt2"):
        """
        :param device: device to use
        :param max_len: maximal length for Tokenizer
        :param pretrained_model_path: model to load for fine-tuning
        """
        super().__init__()
        try:
            self.model = GPT2LMHeadModel.from_pretrained(pretrained_model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_path, model_max_length=max_len)
        except OSError:
            print("Could not load pretrained model")
        self.device = device
        self.max_len = max_len
        self.max_len_model = max_len_model
        self.path = pretrained_model_path
        self.ngram = ngram

    def reinitialize(self, ngrams):
        """
        Reloads model and tokenizer
        """
        try:
            self.model = GPT2LMHeadModel.from_pretrained(self.path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.path, model_max_length=self.max_len)
        except OSError:
            print("Could not load pretrained model")

        self.ngram = ngrams

    def generate(self, sentence):
        sentence_enc = self.tokenizer.encode(sentence, return_tensors='pt') 
        output = self.model.generate(sentence_enc, max_new_tokens=self.max_len_model, num_beams=2, no_repeat_ngram_size=self.ngram, early_stopping=True)
        try:
            return self.tokenizer.decode(output[0])
        except:
            return ""

    def forward(self, x, y):
        """
        :param x: input text
        :param y: text with description cut
        :return: tensor of output data
        """
        
        tokenized = self.tokenizer(x, padding=True, truncation=True, return_tensors="pt").to(self.device)  # токенизируем
        tokenized_nodesc = self.tokenizer(y, padding=True, truncation=True, return_tensors="pt").to(self.device)
    
        input_ids = tokenized['input_ids']
        input_ids_nodesc = tokenized['input_ids']
           
        for i in range(input_ids.size(0)):
            input_ids[i][:len(input_ids_nodesc[i])] = self.tokenizer.pad_token_id
    
        return self.model(**tokenized, labels=input_ids)