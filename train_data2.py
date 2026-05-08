import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import get_model
from utils import (
    analyze_error_samples, analyze_error_distribution,
    count_parameters, measure_inference_time,
    plot_error_distribution, plot_training_history, plot_confusion_matrix,
    print_error_samples, print_error_distribution,
    print_parameter_stats, print_inference_stats
)

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data2')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures_data2')
os.makedirs(FIGURES_DIR, exist_ok=True)

# 加载类别映射
with open(os.path.join(DATA_DIR, 'category_map.json'), 'r', encoding='utf-8') as f:
    category_map = json.load(f)

# 创建反向映射（索引到类别名）
index_to_category = {v-1: k for k, v in category_map.items()}  # 转换为0-based索引
NUM_CLASSES = len(category_map)

print(f"Number of classes: {NUM_CLASSES}")
print(f"Categories: {list(category_map.keys())}")


def load_data():
    """加载数据"""
    print(f"\n{'='*80}")
    print("Loading News dataset from data2...")
    print(f"{'='*80}")
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # 显示类别分布
    print("\nTrain set category distribution:")
    print(train_df['Class Index'].value_counts().sort_index())
    
    return train_df, val_df, test_df


def tokenize_data(train_df, val_df, test_df, tokenizer, max_length):
    """分词数据"""
    print(f"\n{'='*80}")
    print("Tokenizing dataset...")
    print(f"{'='*80}")
    
    def tokenize_function(examples):
        # 合并 Title 和 Description
        texts = [f"{title} {desc}" for title, desc in zip(examples['Title'], examples['Description'])]
        return tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    
    # 转换 Class Index 为 0-based (从 1-20 转换为 0-19)
    train_df['label'] = train_df['Class Index'] - 1
    val_df['label'] = val_df['Class Index'] - 1
    test_df['label'] = test_df['Class Index'] - 1
    
    # 转换为 Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df[['Title', 'Description', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['Title', 'Description', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['Title', 'Description', 'label']])
    
    # 分词
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # 移除不需要的列
    train_dataset = train_dataset.remove_columns(['Title', 'Description'])
    val_dataset = val_dataset.remove_columns(['Title', 'Description'])
    test_dataset = test_dataset.remove_columns(['Title', 'Description'])
    
    # 设置格式
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    print("Tokenization completed!")
    
    return train_dataset, val_dataset, test_dataset


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        loss, logits = model(input_ids, attention_mask, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            loss, logits = model(input_ids, attention_mask, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy, all_labels, all_preds


def train_and_evaluate(model_name, tokenizer, train_dataset, val_dataset, test_dataset, epochs=50, **model_kwargs):
    """训练和评估模型"""
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    LEARNING_RATE = 1e-5
    EARLY_STOP_PATIENCE = 3  # 早停耐心值
    
    print(f"\n{'='*80}")
    print(f"Training Model: {model_name}")
    print(f"{'='*80}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Length: {MAX_LENGTH}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Early Stop Patience: {EARLY_STOP_PATIENCE}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"{'='*80}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 为纯 Transformer 和 GRU+Attention 模型添加词表大小参数
    if model_name in ['transformer', 'gru_attention']:
        model_kwargs['vocab_size'] = tokenizer.vocab_size
    
    # 创建模型
    model = get_model(model_name, num_labels=NUM_CLASSES, **model_kwargs).to(device)
    
    # 打印模型参数
    param_stats = count_parameters(model)
    print_parameter_stats(param_stats)
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_accuracy = 0
    best_model_dir = os.path.join(BASE_DIR, f'{model_name}_data2_model')
    os.makedirs(best_model_dir, exist_ok=True)
    
    # 早停计数器
    patience_counter = 0
    
    print(f"\n{'='*80}")
    print("Starting Training...")
    print(f"{'='*80}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 80)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_labels, val_preds = evaluate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), os.path.join(best_model_dir, 'model.pth'))
            tokenizer.save_pretrained(best_model_dir)
            print(f"Best model saved! Val Acc: {val_acc:.4f}")
            # 重置早停计数器
            patience_counter = 0
        else:
            # 验证准确率没有提升，增加早停计数器
            patience_counter += 1
            print(f"Early stop counter: {patience_counter}/{EARLY_STOP_PATIENCE}")
            
            # 检查是否触发早停
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered!")
                break
    
    print(f"\n{'='*80}")
    print("Training Completed!")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"Total Epochs Trained: {len(history['train_loss'])}")
    print(f"{'='*80}")
    
    # 加载最佳模型
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(best_model_dir, 'model.pth')))
    
    # 测试集评估
    test_loss, test_acc, test_labels, test_preds = evaluate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # 将测试集结果添加到 history
    history['test_loss'] = [test_loss] * len(history['train_loss'])
    history['test_acc'] = [test_acc] * len(history['train_acc'])
    
    # 绘制训练历史图
    history_path = os.path.join(FIGURES_DIR, f'{model_name}_training_history.png')
    plot_training_history(history, history_path)
    
    # 准备类别名称
    target_names = [index_to_category[i] for i in range(NUM_CLASSES)]
    
    # 测试集分类报告
    print(f"\n{'='*80}")
    print("Test Set - Classification Report:")
    print(f"{'='*80}")
    test_report = classification_report(test_labels, test_preds, target_names=target_names, zero_division=0)
    print(test_report)
    
    # 测试集混淆矩阵
    print(f"\n{'='*80}")
    print("Test Set - Confusion Matrix:")
    print(f"{'='*80}")
    test_cm = confusion_matrix(test_labels, test_preds)
    print(test_cm)
    
    # 绘制测试集混淆矩阵
    test_cm_path = os.path.join(FIGURES_DIR, f'{model_name}_test_confusion_matrix.png')
    plot_confusion_matrix(
        test_labels, test_preds, target_names, test_cm_path,
        title=f'{model_name} - Test Set Confusion Matrix'
    )
    
    # 推理时间统计
    print(f"\n{'='*80}")
    print("Measuring Inference Time")
    print(f"{'='*80}")
    inference_stats = measure_inference_time(model, test_loader, device, num_batches=10)
    print_inference_stats(inference_stats)
    
    # 保存结果到文件
    result_file = os.path.join(FIGURES_DIR, f'{model_name}_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"{model_name} Model Training Results (Data2)\n")
        f.write("="*80 + "\n\n")
        
        f.write("Model Parameters:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Parameters: {param_stats['total']:,}\n")
        f.write(f"Trainable Parameters: {param_stats['trainable']:,}\n")
        f.write(f"Non-trainable Parameters: {param_stats['non_trainable']:,}\n\n")
        
        f.write("Hyperparameters:\n")
        f.write("-"*80 + "\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Max Length: {MAX_LENGTH}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Number of Classes: {NUM_CLASSES}\n\n")
        
        f.write("Training History:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Epoch':<10}{'Train Loss':<15}{'Train Acc':<15}{'Val Loss':<15}{'Val Acc':<15}\n")
        f.write("-"*80 + "\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1:<10}{history['train_loss'][i]:<15.4f}{history['train_acc'][i]:<15.4f}"
                   f"{history['val_loss'][i]:<15.4f}{history['val_acc'][i]:<15.4f}\n")
        f.write("\n")
        
        f.write("Test Results:\n")
        f.write("-"*80 + "\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        
        f.write("Test Set - Classification Report:\n")
        f.write("-"*80 + "\n")
        f.write(test_report + "\n\n")
        
        f.write("Test Set - Confusion Matrix:\n")
        f.write("-"*80 + "\n")
        f.write(str(test_cm) + "\n\n")
        
        f.write("Inference Time:\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean: {inference_stats['mean']:.4f}s\n")
        f.write(f"Std: {inference_stats['std']:.4f}s\n")
        f.write(f"Throughput: {1/inference_stats['mean']:.2f} samples/s\n")
    
    print(f"\nResults saved to: {result_file}")
    
    return {
        'param_stats': param_stats,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'history': history
    }


def main():
    """主函数"""
    # 定义所有可用的模型
    all_models = {
        '1': ('bert', {}),
        '2': ('bert_gru_attention', {'hidden_size': 256, 'num_layers': 1, 'dropout': 0.1}),
        '3': ('bert_transformer', {'num_heads': 8, 'num_layers': 1, 'dropout': 0.1}),
        '4': ('bert_fpn', {}),
        '5': ('bert_gru_attention_frozen', {'hidden_size': 256, 'num_layers': 1, 'dropout': 0.3, 'freeze_layers': 8}),
        '6': ('bert_transformer_frozen', {'num_heads': 8, 'num_layers': 1, 'dropout': 0.1, 'freeze_layers': 8}),
        '7': ('transformer', {'d_model': 512, 'num_heads': 8, 'num_layers': 6, 'd_ff': 2048, 'max_len': 128, 'dropout': 0.1}),
        '8': ('gru_attention', {'embedding_dim': 300, 'hidden_size': 256, 'num_layers': 1, 'dropout': 0.3, 'bidirectional': True})
    }
    
    # 显示模型选择菜单
    print("\n" + "="*80)
    print("Available Models for Data2 Training")
    print("="*80)
    print("1. BERT Baseline")
    print("2. BERT + GRU + Attention")
    print("3. BERT + Transformer")
    print("4. BERT + FPN")
    print("5. BERT + GRU + Attention (Frozen 8 Layers)")
    print("6. BERT + Transformer (Frozen 8 Layers)")
    print("7. Pure Transformer")
    print("8. Pure GRU + Attention")
    print("9. Train All Models")
    print("="*80)
    
    # 获取用户选择
    choice = input("\nPlease select a model to train (1-9): ").strip()
    
    # 根据选择确定要训练的模型
    if choice == '9':
        models_to_train = [all_models['1'], all_models['2'], all_models['3'], all_models['4'], all_models['5'], all_models['6'], all_models['7'], all_models['8']]
    elif choice in all_models:
        models_to_train = [all_models[choice]]
    else:
        print("Invalid choice. Please run the script again and select a valid option (1-9).")
        return
    
    # 加载数据
    train_df, val_df, test_df = load_data()
    
    # 初始化 tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    # 分词数据
    train_dataset, val_dataset, test_dataset = tokenize_data(
        train_df, val_df, test_df, tokenizer, max_length=128
    )
    
    # 存储所有模型的结果
    all_results = {}
    
    # 训练和评估每个模型
    for model_name, model_kwargs in models_to_train:
        print(f"\n{'='*80}")
        print(f"Training Model: {model_name}")
        print(f"{'='*80}")
        
        train_results = train_and_evaluate(
            model_name, tokenizer, train_dataset, val_dataset, test_dataset,
            epochs=50, **model_kwargs
        )
        
        # 保存结果
        all_results[model_name] = {
            'train_results': train_results
        }
    
    # 如果训练了多个模型，进行比较
    if len(models_to_train) > 1:
        print(f"\n{'='*80}")
        print("Model Comparison")
        print(f"{'='*80}")
        print(f"{'Model':<25}{'Test Accuracy':<20}{'Parameters':<20}")
        print("-"*80)
        for model_name, results in all_results.items():
            test_acc = results['train_results']['test_accuracy']
            params = results['train_results']['param_stats']['total']
            print(f"{model_name:<25}{test_acc:<20.4f}{params:<20,}")


if __name__ == '__main__':
    main()
