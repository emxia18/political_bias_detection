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
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        if attention_bias is not None:
            if attention_bias.dim() == 2: 
                attention_bias = attention_bias[:, 0].unsqueeze(-1) 

            attention_bias = attention_bias.expand(-1, cls_embedding.shape[1])
            cls_embedding = cls_embedding * (1 + attention_bias)

        logits = self.classifier(cls_embedding)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}
