import torch
import torch.nn as nn
from transformers import BertForSequenceClassification


class BertBaselineFrozen(nn.Module):
    """
    BERT 基础模型（冻结部分层）
    BERT baseline model with frozen layers
    """
    
    def __init__(self, num_labels=4, model_name='bert-base-uncased', freeze_layers=8):
        """
        初始化 BERT 基础模型
        Initialize BERT baseline model
        
        Args:
            num_labels: 分类数量 / Number of classes
            model_name: 预训练模型名称 / Pretrained model name
            freeze_layers: 冻结的层数 / Number of frozen layers (default: 8)
        """
        super(BertBaselineFrozen, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.freeze_layers = freeze_layers
        
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # 冻结 BERT 的底层
        self._freeze_bert_layers()
    
    def _freeze_bert_layers(self):
        """
        冻结 BERT 的底层
        Freeze bottom layers of BERT
        """
        # BERT base 有 12 层 encoder
        # 冻结前 freeze_layers 层，只训练后 (12 - freeze_layers) 层
        for param in self.bert.bert.embeddings.parameters():
            param.requires_grad = False
        
        for i in range(self.freeze_layers):
            for param in self.bert.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        print(f"Frozen {self.freeze_layers} BERT encoder layers")
        print(f"Training top {12 - self.freeze_layers} BERT encoder layers + classifier")
    
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
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        if labels is not None:
            return outputs.loss, outputs.logits
        else:
            return outputs.logits
    
    def get_num_parameters(self):
        """
        获取模型参数量
        Get number of model parameters
        
        Returns:
            总参数量 / Total number of parameters
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
