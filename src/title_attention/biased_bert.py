import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertForSequenceClassification

class BiasedDistilBERT(DistilBertForSequenceClassification):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3):
        super().__init__(config=self.config)
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, attention_bias=None):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask, output_attentions=True)
        hidden_states = outputs.last_hidden_state

        all_attention = outputs.attentions

        if attention_bias is not None:
            biased_attention = [
                attn + attention_bias.unsqueeze(1).unsqueeze(1) for attn in all_attention
            ]
        else:
            biased_attention = all_attention

        weighted_hidden_states = (hidden_states * attention_bias.unsqueeze(-1)).sum(dim=1) 

        logits = self.classifier(weighted_hidden_states)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits, "biased_attention": biased_attention}
