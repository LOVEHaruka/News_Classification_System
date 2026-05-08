import torch
import torch.nn as nn
from transformers import BertModel


class AttentionLayer(nn.Module):
    """
    注意力层
    Attention layer
    """
    
    def __init__(self, hidden_size):
        """
        初始化注意力层
        Initialize attention layer
        
        Args:
            hidden_size: 隐藏层维度
        """
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, hidden_states):
        """
        前向传播
        Forward pass
        
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
        
        Returns:
            注意力权重和加权后的输出
        """
        attention_weights = self.attention(hidden_states)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)  # [batch_size, seq_len, 1]
        
        weighted_output = torch.sum(hidden_states * attention_weights, dim=1)  # [batch_size, hidden_size]
        
        return weighted_output, attention_weights


class BertGRUAttentionFrozen(nn.Module):
    """
    BERT + GRU + Attention 模型（冻结8层）
    BERT + GRU + Attention model with 8 frozen layers
    """
    
    def __init__(self, num_labels=4, model_name='bert-base-uncased', hidden_size=256, num_layers=1, dropout=0.3, freeze_layers=8):
        """
        初始化 BERT+GRU+Attention 模型（冻结8层）
        Initialize BERT+GRU+Attention model with 8 frozen layers
        
        Args:
            num_labels: 分类数量 / Number of classes
            model_name: 预训练模型名称 / Pretrained model name
            hidden_size: GRU 隐藏层维度 / GRU hidden size
            num_layers: GRU 层数 / Number of GRU layers
            dropout: Dropout 概率 / Dropout probability
            freeze_layers: 冻结的 BERT 层数 / Number of frozen BERT layers
        """
        super(BertGRUAttentionFrozen, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.freeze_layers = freeze_layers
        
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # 冻结 BERT 的底层
        self._freeze_bert_layers()
        
        # GRU 层
        self.gru = nn.GRU(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 注意力层
        self.attention = AttentionLayer(hidden_size * 2)  # 双向 GRU
        
        # 分类层
        self.classifier = nn.Linear(hidden_size * 2, num_labels)
    
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
        print(f"Training top {12 - self.freeze_layers} BERT encoder layers + GRU + Attention + Classifier")
    
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
        sequence_output = self.dropout(sequence_output)
        
        # GRU 编码
        gru_output, _ = self.gru(sequence_output)  # [batch_size, seq_len, hidden_size*2]
        
        # 注意力机制
        attended_output, attention_weights = self.attention(gru_output)  # [batch_size, hidden_size*2]
        
        # 分类
        logits = self.classifier(attended_output)  # [batch_size, num_labels]
        
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
