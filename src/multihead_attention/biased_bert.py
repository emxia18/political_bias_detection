import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertForSequenceClassification, DistilBertConfig

class BiasedDistilBERT(DistilBertForSequenceClassification):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3, hidden_dim=512, num_heads=4, dropout_rate=0.3):
        config = DistilBertConfig.from_pretrained(model_name, num_labels=num_labels)
        super().__init__(config)

        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.attention = nn.MultiheadAttention(embed_dim=self.distilbert.config.hidden_size, num_heads=num_heads, batch_first=True)
        
        self.fc1 = nn.Linear(self.distilbert.config.hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_labels)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.residual = nn.Linear(self.distilbert.config.hidden_size, hidden_dim) 

    def forward(self, input_ids, attention_mask, labels=None, attention_bias=None):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask, output_attentions=True)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        if attention_bias is not None:
            if attention_bias.dim() == 3:
                attention_bias = attention_bias.squeeze(-1) 
            
            attention_bias = attention_bias.expand(-1, cls_embedding.shape[1])
            cls_embedding = cls_embedding * attention_bias

        attn_output, _ = self.attention(cls_embedding.unsqueeze(1), cls_embedding.unsqueeze(1), cls_embedding.unsqueeze(1))
        attn_output = attn_output.squeeze(1)

        residual = self.residual(attn_output)
        x = self.fc1(attn_output)
        x = self.activation(x)
        x = self.dropout(x)

        x = x + residual
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.fc3(x)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}
