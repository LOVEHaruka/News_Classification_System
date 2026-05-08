import torch
import torch.nn as nn
from transformers import BertModel


class BertTransformerFrozen(nn.Module):
    """
    BERT + Transformer 模型（冻结8层）
    BERT + Transformer model with 8 frozen layers
    """
    
    def __init__(self, num_labels=4, model_name='bert-base-uncased', num_heads=8, num_layers=1, dropout=0.1, freeze_layers=8):
        """
        初始化 BERT+Transformer 模型（冻结8层）
        Initialize BERT+Transformer model with 8 frozen layers
        
        Args:
            num_labels: 分类数量 / Number of classes
            model_name: 预训练模型名称 / Pretrained model name
            num_heads: Transformer 注意力头数 / Number of attention heads
            num_layers: Transformer 层数 / Number of transformer layers
            dropout: Dropout 概率 / Dropout probability
            freeze_layers: 冻结的 BERT 层数 / Number of frozen BERT layers
        """
        super(BertTransformerFrozen, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.freeze_layers = freeze_layers
        
        self.bert = BertModel.from_pretrained(model_name)
        bert_hidden_size = self.bert.config.hidden_size
        
        # 冻结 BERT 的底层
        self._freeze_bert_layers()
        
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
    
    def _freeze_bert_layers(self):
        """
        冻结 BERT 的底层
        Freeze bottom layers of BERT
        """
        # BERT base 有 12 层 encoder
        # 冻结前 freeze_layers 层，只训练后 (12 - freeze_layers) 层
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        for i in range(self.freeze_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        print(f"Frozen {self.freeze_layers} BERT encoder layers")
        print(f"Training top {12 - self.freeze_layers} BERT encoder layers + Transformer + Classifier")
    
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
