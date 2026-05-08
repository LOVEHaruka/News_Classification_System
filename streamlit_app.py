import streamlit as st
import torch
import numpy as np
import re
from transformers import BertTokenizer
import os
from huggingface_hub import snapshot_download

# 模型仓库地址
MODEL_REPO = 'LOVEHaruka/News_Classification_System_bert_gru_attention_model'
MODEL_PATH = snapshot_download(repo_id=MODEL_REPO)

# 类别映射（4分类）
CLASS_4_MAP = {
    0: 'World',
    1: 'Sports',
    2: 'Business',
    3: 'Science & Technology'
}

class BERTTransformerModel(torch.nn.Module):
    """BERT + Transformer 模型"""
    def __init__(self, num_classes):
        super(BERTTransformerModel, self).__init__()
        # 加载BERT模型
        from transformers import BertModel
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Transformer编码器
        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=768,  # BERT输出维度
                nhead=8,
                dim_feedforward=3072
            ),
            num_layers=1
        )
        
        # 分类器
        self.classifier = torch.nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # BERT编码
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = bert_output.last_hidden_state
        
        # Transformer处理
        transformer_output = self.transformer_encoder(last_hidden_state)
        cls_output = transformer_output[:, 0, :]  # 取[CLS] token的输出
        
        # 分类
        output = self.classifier(cls_output)
        return output

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

# 加载模型
@st.cache_resource
def load_model():
    """加载模型和tokenizer"""
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 加载权重文件查看分类数
    model_path_full = os.path.join(MODEL_PATH, 'model.pth')
    state_dict = torch.load(model_path_full, map_location=torch.device('cpu'))
    
    # 确定分类数
    if 'classifier.weight' in state_dict:
        num_classes = state_dict['classifier.weight'].shape[0]
    else:
        num_classes = 4
    
    # 创建模型
    model = BERTTransformerModel(num_classes=num_classes)
    
    # 加载权重
    model.load_state_dict(state_dict)
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
