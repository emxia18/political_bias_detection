import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertForSequenceClassification, DistilBertConfig

class BiasedDistilBERT(DistilBertForSequenceClassification):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3):
        config = DistilBertConfig.from_pretrained(model_name, num_labels=num_labels)
        super().__init__(config)
        
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, attention_bias=None):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask, output_attentions=True)
        hidden_states = outputs.last_hidden_state

        if attention_bias is not None:
            hidden_states = hidden_states * attention_bias.unsqueeze(-1) 

        weighted_output = hidden_states.mean(dim=1)

        logits = self.classifier(weighted_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}
