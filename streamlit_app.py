import streamlit as st
import torch
import numpy as np
import re
import os

try:
    from transformers import BertTokenizer
except ImportError:
    from transformers.models.bert import BertTokenizer

# 模型仓库地址
MODEL_REPO = 'LOVEHaruka/News_Classification_System_bert_gru_attention_model'

# 类别映射（4分类）
CLASS_4_MAP = {
    0: 'World',
    1: 'Sports',
    2: 'Business',
    3: 'Science & Technology'
}

class AttentionLayer(torch.nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1)
        )
    
    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_output = torch.sum(hidden_states * attention_weights, dim=1)
        return weighted_output, attention_weights

class BertGRUAttention(torch.nn.Module):
    def __init__(self, num_labels=4, model_name='bert-base-uncased', hidden_size=256, num_layers=1, dropout=0.3):
        super(BertGRUAttention, self).__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.attention = AttentionLayer(hidden_size * 2)
        self.classifier = torch.nn.Linear(hidden_size * 2, num_labels)
    
    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        gru_output, _ = self.gru(sequence_output)
        attended_output, _ = self.attention(gru_output)
        logits = self.classifier(attended_output)
        return logits

def clean_text(text):
    """清洗文本"""
    if not text:
        return ""
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def download_model_file():
    """下载模型文件，处理可能的LFS问题"""
    from huggingface_hub import hf_hub_download
    import requests
    
    cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
    local_model_path = os.path.join(cache_dir, 'model.pth')
    
    # 首先检查本地缓存是否存在
    if os.path.exists(local_model_path) and os.path.getsize(local_model_path) > 0:
        print(f"Using cached model: {local_model_path}")
        return local_model_path
    
    # 方法1: 使用 hf_hub_download
    try:
        model_path_full = hf_hub_download(
            repo_id=MODEL_REPO,
            filename='model.pth'
        )
        if os.path.exists(model_path_full) and os.path.getsize(model_path_full) > 0:
            return model_path_full
    except Exception as e:
        print(f"hf_hub_download failed: {e}")
    
    # 方法2: 尝试直接HTTP下载
    try:
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/model.pth"
        os.makedirs(cache_dir, exist_ok=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        if os.path.exists(local_model_path) and os.path.getsize(local_model_path) > 0:
            return local_model_path
    except Exception as e:
        print(f"HTTP download failed: {e}")
    
    return None

# 加载模型
@st.cache_resource
def load_model():
    """加载模型和tokenizer"""
    model_path_full = download_model_file()
    
    if model_path_full is None or not os.path.exists(model_path_full):
        error_msg = f"""
        Model file 'model.pth' not found!
        
        Possible reasons:
        1. The file was uploaded using Git LFS but not properly downloaded
        2. The repository doesn't have the model file accessible
        
        Please check: https://huggingface.co/{MODEL_REPO}
        """
        raise FileNotFoundError(error_msg)
    
    model_path = os.path.dirname(model_path_full)
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 加载权重文件查看分类数
    state_dict = torch.load(model_path_full, map_location=torch.device('cpu'))
    
    # 确定分类数
    if 'classifier.weight' in state_dict:
        num_classes = state_dict['classifier.weight'].shape[0]
    else:
        num_classes = 4
    
    # 创建模型
    model = BertGRUAttention(num_labels=num_classes)
    
    # 加载权重（忽略不匹配的键）
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # 移除不匹配的键
        state_dict = {k: v for k, v in state_dict.items() if "position_ids" not in k}
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, tokenizer, num_classes

def predict(text, model, tokenizer, num_classes):
    """预测文本类别"""
    # 清洗文本
    cleaned_text = clean_text(text)
    
    # 分词
    inputs = tokenizer(
        cleaned_text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 推理
    with torch.no_grad():
        output = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
    
    # 获取预测结果
    probabilities = torch.softmax(output, dim=1).squeeze().numpy()
    predicted_class = np.argmax(probabilities)
    
    # 获取类别名称
    class_name = CLASS_4_MAP.get(predicted_class, f'Class {predicted_class}')
    
    return predicted_class, class_name, probabilities[predicted_class]

def main():
    """主函数"""
    st.title("News Text Classifier")
    st.markdown("Enter news text to get classification results")
    
    # 加载模型
    with st.spinner("Loading model..."):
        model, tokenizer, num_classes = load_model()
    
    st.success("Model loaded successfully!")
    
    # 输入文本
    user_input = st.text_area("Enter news text:", height=200)
    
    if st.button("Classify"):
        if not user_input.strip():
            st.error("Please enter valid news text")
        else:
            # 4分类模型预测
            st.subheader("4-class Model Prediction")
            try:
                pred_class, class_name, score = predict(user_input, model, tokenizer, num_classes)
                st.success(f"Predicted class: **{class_name}** (Class {pred_class})")
                st.info(f"Confidence: {score:.4f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

if __name__ == '__main__':
    main()
