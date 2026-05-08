from .bert import BertBaseline
from .bert_gru_attention import BertGRUAttention
from .bert_transformer import BertTransformer
from .bert_gru_attention_frozen import BertGRUAttentionFrozen
from .bert_transformer_frozen import BertTransformerFrozen
from .bert_fpn import BertFPN
from .bert_fpn_two_layer import BertFPNTwoLayer
from .bert_fpn_gated import BertFPNGated
from .transformer import PureTransformer
from .gru_attention import GRUAttention


def get_model(model_name, num_labels=4, **kwargs):
    """
    模型工厂函数
    Model factory function
    
    Args:
        model_name: 模型名称 / Model name ('bert', 'bert_gru_attention', 'bert_transformer', 'bert_gru_attention_frozen', 'bert_transformer_frozen', 'bert_fpn', 'transformer', 'gru_attention')
        num_labels: 分类数量 / Number of classes
        **kwargs: 其他模型参数 / Other model parameters
    
    Returns:
        模型实例 / Model instance
    """
    models = {
        'bert': BertBaseline,
        'bert_gru_attention': BertGRUAttention,
        'bert_transformer': BertTransformer,
        'bert_gru_attention_frozen': BertGRUAttentionFrozen,
        'bert_transformer_frozen': BertTransformerFrozen,
        'bert_fpn': BertFPN,
        'bert_fpn_two_layer': BertFPNTwoLayer,
        'bert_fpn_gated': BertFPNGated,
        'transformer': PureTransformer,
        'gru_attention': GRUAttention
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name](num_labels=num_labels, **kwargs)
