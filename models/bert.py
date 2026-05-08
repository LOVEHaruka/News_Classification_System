import torch
import torch.nn as nn
from transformers import BertForSequenceClassification


class BertBaseline(nn.Module):
    """
    BERT 基础模型
    BERT baseline model
    """
    
    def __init__(self, num_labels=4, model_name='bert-base-uncased'):
        """
        初始化 BERT 基础模型
        Initialize BERT baseline model
        
        Args:
            num_labels: 分类数量 / Number of classes
            model_name: 预训练模型名称 / Pretrained model name
        """
        super(BertBaseline, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    
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
        return sum(p.numel() for p in self.parameters())
