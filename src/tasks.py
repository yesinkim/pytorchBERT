import torch.nn as nn
from torch import Tensor


class MaskedLanguageModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int
        ) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, bert_output: Tensor) -> Tensor:
        return self.softmax(self.linear(bert_output))


class NextSentencePrediction(nn.Module):
    def __init__(
        self,
        hidden_dim: int
        ) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 2)
        self.softmax = nn.LogSigmoid(dim=-1)
    
    def forward(self, bert_output: Tensor) -> Tensor:
        return self.softmax(self.linear(bert_output[:, 0]))