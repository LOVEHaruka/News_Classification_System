import torch
import torch.nn as nn
from transformers import BertModel


class BertTransformer(nn.Module):
    """
    BERT + Transformer 模型
    BERT + Transformer model
    """
    
    def __init__(self, num_labels=4, model_name='bert-base-uncased', num_heads=8, num_layers=2, dropout=0.1):
        """
        初始化 BERT+Transformer 模型
        Initialize BERT+Transformer model
        
        Args:
            num_labels: 分类数量 / Number of classes
            model_name: 预训练模型名称 / Pretrained model name
            num_heads: Transformer 注意力头数 / Number of attention heads
            num_layers: Transformer 层数 / Number of transformer layers
            dropout: Dropout 概率 / Dropout probability
        """
        super(BertTransformer, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.bert = BertModel.from_pretrained(model_name)
        bert_hidden_size = self.bert.config.hidden_size
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bert_hidden_size,
            nhead=num_heads,
            dim_feedforward=bert_hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 分类层
        self.classifier = nn.Linear(bert_hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播
        Forward pass
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            labels: 标签（训练时使用）/ Labels (used during training)
        
        Returns:
            如果 labels 不为 None，返回 loss 和 logits
            否则只返回 logits
        """
        # BERT 编码
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 获取 BERT 输出
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Transformer 编码
        transformer_output = self.transformer_encoder(sequence_output)  # [batch_size, seq_len, hidden_size]
        
        # 使用 [CLS] token 的输出进行分类
        cls_output = transformer_output[:, 0, :]  # [batch_size, hidden_size]
        cls_output = self.dropout(cls_output)
        
        # 分类
        logits = self.classifier(cls_output)  # [batch_size, num_labels]
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
    
    def get_num_parameters(self):
        """
        获取模型参数量
        Get number of model parameters
        
        Returns:
            总参数量 / Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())
