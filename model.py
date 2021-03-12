# 3rd Party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import DistilBertForQuestionAnswering

class Discriminator(nn.Module):
    def __init__(self, num_classes=6, input_size=768 * 2,
                 hidden_size=768, num_layers=3, dropout=0.1):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            hidden_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(), nn.Dropout(dropout)
            ))
        hidden_layers.append(nn.Linear(hidden_size, num_classes))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        # forward pass
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
        logits = self.hidden_layers[-1](x)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob

class EnsembleOutput():
    def __init__(self, start_logits, end_logits):
        self.start_logits = start_logits
        self.end_logits = end_logits

class Ensemble(nn.Module):
    def __init__(self, device='cpu'):
        super(Ensemble, self).__init__()

        self.device = device
        self.models = []
        self.weights = []

    def add_pretrained_model(self, path, weight):
        model = DistilBertForQuestionAnswering.from_pretrained(path)
        self.models.append(model)
        self.weights.append(weight)

    def forward(self, input_ids, attention_mask):
        """
        Return:
          EnsembleOutput object:
            start_logits: Tensor of size (batch_size, num_models, max_length)
            end_logits: Tensor of size (batch_size, num_models, max_length)
        """

        start_logits = []
        end_logits = []
        
        for model in self.models:

            outputs = model(input_ids, attention_mask=attention_mask)

            # Forward
            start_logits.append(outputs.start_logits.unsqueeze(dim=1))
            end_logits.append(outputs.end_logits.unsqueeze(dim=1))

        start_logits = torch.cat(start_logits, dim=1)
        end_logits = torch.cat(end_logits, dim=1)

        return EnsembleOutput(start_logits, end_logits)
