import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置编码
    Positional Encoding
    """
    
    def __init__(self, d_model, max_len=512, dropout=0.1):
        """
        初始化位置编码
        Initialize positional encoding
        
        Args:
            d_model: 模型维度 / Model dimension
            max_len: 最大序列长度 / Maximum sequence length
            dropout: Dropout 概率 / Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播
        Forward pass
        
        Args:
            x: 输入张量 / Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            添加位置编码后的张量 / Tensor with positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PureTransformer(nn.Module):
    """
    纯 Transformer 模型
    Pure Transformer model
    """
    
    def __init__(self, num_labels=4, vocab_size=30522, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, max_len=512, dropout=0.1):
        """
        初始化纯 Transformer 模型
        Initialize pure Transformer model
        
        Args:
            num_labels: 分类数量 / Number of classes
            vocab_size: 词表大小 / Vocabulary size
            d_model: 模型维度 / Model dimension
            num_heads: 注意力头数 / Number of attention heads
            num_layers: Transformer 层数 / Number of transformer layers
            d_ff: 前馈网络维度 / Feed-forward network dimension
            max_len: 最大序列长度 / Maximum sequence length
            dropout: Dropout 概率 / Dropout probability
        """
        super(PureTransformer, self).__init__()
        self.num_labels = num_labels
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 分类层
        self.classifier = nn.Linear(d_model, num_labels)
        self.dropout = nn.Dropout(dropout)
        
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
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
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
        x = self.embedding(input_ids)  # [batch_size, seq_len, d_model]
        x = x * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer 编码
        if attention_mask is not None:
            # 将注意力掩码转换为 Transformer 需要的格式
            # attention_mask: 1 表示有效 token，0 表示 padding
            # Transformer 需要 False 表示 padding，True 表示有效 token
            src_key_padding_mask = (attention_mask == 0)
            transformer_output = self.transformer_encoder(
                x, 
                src_key_padding_mask=src_key_padding_mask
            )
        else:
            transformer_output = self.transformer_encoder(x)
        
        # 使用第一个 token 的输出进行分类
        cls_output = transformer_output[:, 0, :]  # [batch_size, d_model]
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
