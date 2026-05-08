import torch
import torch.nn as nn


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
    
    def forward(self, hidden_states, attention_mask=None):
        """
        前向传播
        Forward pass
        
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        
        Returns:
            注意力权重和加权后的输出
        """
        attention_scores = self.attention(hidden_states)  # [batch_size, seq_len, 1]
        
        if attention_mask is not None:
            # 将注意力掩码扩展到注意力分数的维度
            mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        
        weighted_output = torch.sum(hidden_states * attention_weights, dim=1)  # [batch_size, hidden_size]
        
        return weighted_output, attention_weights


class GRUAttention(nn.Module):
    """
    纯 GRU + Attention 模型
    Pure GRU + Attention model
    """
    
    def __init__(self, num_labels=4, vocab_size=30522, embedding_dim=300, 
                 hidden_size=256, num_layers=1, dropout=0.3, bidirectional=True):
        """
        初始化纯 GRU+Attention 模型
        Initialize pure GRU+Attention model
        
        Args:
            num_labels: 分类数量 / Number of classes
            vocab_size: 词表大小 / Vocabulary size
            embedding_dim: 词嵌入维度 / Embedding dimension
            hidden_size: GRU 隐藏层维度 / GRU hidden size
            num_layers: GRU 层数 / Number of GRU layers
            dropout: Dropout 概率 / Dropout probability
            bidirectional: 是否使用双向 GRU / Whether to use bidirectional GRU
        """
        super(GRUAttention, self).__init__()
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # GRU 层
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力层
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = AttentionLayer(gru_output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 分类层
        self.classifier = nn.Linear(gru_output_size, num_labels)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化模型参数
        Initialize model weights
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        前向传播
        Forward pass
        
        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签（训练时使用）/ Labels (used during training)
        
        Returns:
            如果 labels 不为 None，返回 loss 和 logits
            否则只返回 logits
        """
        # 词嵌入
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # GRU 编码
        gru_output, _ = self.gru(embedded)  # [batch_size, seq_len, hidden_size*2]
        gru_output = self.dropout(gru_output)
        
        # 注意力机制
        attended_output, attention_weights = self.attention(gru_output, attention_mask)
        
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
