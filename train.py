import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import get_model
from utils import (
    analyze_error_samples, analyze_error_distribution,
    count_parameters, measure_inference_time,
    plot_error_distribution, plot_training_history,
    print_error_samples, print_error_distribution,
    print_parameter_stats, print_inference_stats
)

# 设置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 类别映射
CLASS_NAMES = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
NUM_CLASSES = 4


def load_and_preprocess_data():
    """
    加载和预处理数据
    Load and preprocess data
    
    Returns:
        tokenizer, dataset_tokenized
    """
    print("Loading AG News dataset from local files...")
    
    import pandas as pd
    from datasets import Dataset, DatasetDict
    
    # 读取本地 CSV 文件
    data_dir = os.path.join(BASE_DIR, 'data')
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=0)
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'), header=0)
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=0)
    
    # 重命名列
    train_df = train_df.rename(columns={'Class Index': 'label', 'Title': 'title', 'Description': 'text'})
    val_df = val_df.rename(columns={'Class Index': 'label', 'Title': 'title', 'Description': 'text'})
    test_df = test_df.rename(columns={'Class Index': 'label', 'Title': 'title', 'Description': 'text'})
    
    # 合并标题和文本
    train_df['text'] = train_df['title'] + ' ' + train_df['text']
    val_df['text'] = val_df['title'] + ' ' + val_df['text']
    test_df['text'] = test_df['title'] + ' ' + test_df['text']
    
    # 标签从 1-4 转换为 0-3
    train_df['label'] = train_df['label'] - 1
    val_df['label'] = val_df['label'] - 1
    test_df['label'] = test_df['label'] - 1
    
    # 转换为 Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
    
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=64
        )
    
    print("Tokenizing dataset...")
    dataset_tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    dataset_tokenized.set_format("torch")
    
    return tokenizer, dataset_tokenized


def plot_confusion_matrix(y_true, y_pred, target_names, output_path, title='Confusion Matrix'):
    """
    绘制混淆矩阵
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=target_names,
           yticklabels=target_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {output_path}")
    plt.close()
    
    return cm


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    训练一个epoch
    Train one epoch
    
    Returns:
        avg_loss, accuracy, all_labels, all_preds
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask, labels)
        loss, logits = outputs[0], outputs[1]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_labels, all_preds


def evaluate(model, dataloader, device):
    """
    评估模型
    Evaluate model
    
    Returns:
        avg_loss, accuracy, all_labels, all_preds
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss, logits = outputs[0], outputs[1]
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_labels, all_preds





def train_and_evaluate(model_name, tokenizer, dataset_tokenized, epochs=50, **model_kwargs):
    """
    训练和评估模型
    Train and evaluate model
    
    Args:
        model_name: 模型名称
        tokenizer: 分词器
        dataset_tokenized: 分词后的数据集
        epochs: 训练轮数，默认20
        **model_kwargs: 模型参数
    
    Returns:
        训练结果
    """
    # 超参数
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    EPOCHS = epochs
    LEARNING_RATE = 1e-5
    WARMUP_STEPS = 0
    EARLY_STOP_PATIENCE = 3  # 早停耐心值
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset_tokenized["train"], 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    # 使用本地的验证集和测试集
    val_dataset = dataset_tokenized["validation"]
    test_dataset = dataset_tokenized["test"]
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    print(f"\nTraining set size: {len(dataset_tokenized['train'])}")
    print(f"Validation set size: {len(dataset_tokenized['validation'])}")
    print(f"Test set size: {len(dataset_tokenized['test'])}")
    
    # 创建图表保存目录
    figures_dir = os.path.join(BASE_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 为纯 Transformer 和 GRU+Attention 模型添加词表大小参数
    if model_name in ['transformer', 'gru_attention']:
        model_kwargs['vocab_size'] = tokenizer.vocab_size
    
    # 初始化模型
    print(f"\nLoading {model_name} model...")
    model = get_model(model_name, num_labels=NUM_CLASSES, **model_kwargs)
    model = model.to(device)
    
    # 统计参数量
    param_stats = count_parameters(model)
    print_parameter_stats(param_stats)
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 训练循环
    print(f"\n{'='*80}")
    print("Start Training")
    print(f"{'='*80}")
    print(f"Epochs: {EPOCHS}")
    print(f"Early Stop Patience: {EARLY_STOP_PATIENCE}")
    print(f"{'='*80}")
    
    best_val_accuracy = 0
    # 早停计数器
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 80)
        
        # 训练
        train_loss, train_acc, train_labels, train_preds = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        
        # 验证
        val_loss, val_acc, val_labels, val_preds = evaluate(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        
        # 保存历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            output_dir = os.path.join(BASE_DIR, f'{model_name}_model')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
            tokenizer.save_pretrained(output_dir)
            print(f"Best model saved to: {output_dir}")
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
    
    # 验证集评估
    target_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    
    print(f"\n{'='*80}")
    print("Validation Set - Final Classification Report:")
    print(f"{'='*80}")
    val_report = classification_report(val_labels, val_preds, target_names=target_names)
    print(val_report)
    
    # 验证集混淆矩阵
    print(f"\n{'='*80}")
    print("Validation Set - Confusion Matrix:")
    print(f"{'='*80}")
    val_cm = confusion_matrix(val_labels, val_preds)
    print(val_cm)
    
    # 绘制验证集混淆矩阵
    val_cm_path = os.path.join(figures_dir, f'{model_name}_val_confusion_matrix.png')
    plot_confusion_matrix(
        val_labels, val_preds, target_names, val_cm_path,
        title=f'{model_name} - Validation Set Confusion Matrix'
    )
    
    # 错误分布分析
    error_dist = analyze_error_distribution(val_labels, val_preds, target_names)
    print_error_distribution(error_dist, target_names)
    
    # 绘制错误分布图
    error_dist_path = os.path.join(figures_dir, f'{model_name}_error_distribution.png')
    plot_error_distribution(error_dist, target_names, error_dist_path)
    
    # 错误样本分析
    error_samples = analyze_error_samples(val_dataset, model, tokenizer, device, num_samples=10)
    print_error_samples(error_samples, target_names)
    
    # 测试集评估
    print(f"\n{'='*80}")
    print("Test Set Evaluation")
    print(f"{'='*80}")
    
    # 加载最佳模型
    print("\nLoading best model for test evaluation...")
    best_model_dir = os.path.join(BASE_DIR, f'{model_name}_model')
    model.load_state_dict(torch.load(os.path.join(best_model_dir, 'model.pth')))
    
    # 测试集评估
    test_loss, test_acc, test_labels, test_preds = evaluate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # 将测试集结果添加到 history 中
    history['test_loss'] = [test_loss] * len(history['train_loss'])
    history['test_acc'] = [test_acc] * len(history['train_loss'])
    
    # 绘制训练历史图（包含训练、验证、测试集）
    history_path = os.path.join(figures_dir, f'{model_name}_training_history.png')
    plot_training_history(history, history_path)
    
    # 测试集分类报告
    print(f"\n{'='*80}")
    print("Test Set - Classification Report:")
    print(f"{'='*80}")
    test_report = classification_report(test_labels, test_preds, target_names=target_names)
    print(test_report)
    
    # 测试集混淆矩阵
    print(f"\n{'='*80}")
    print("Test Set - Confusion Matrix:")
    print(f"{'='*80}")
    test_cm = confusion_matrix(test_labels, test_preds)
    print(test_cm)
    
    # 绘制测试集混淆矩阵
    test_cm_path = os.path.join(figures_dir, f'{model_name}_test_confusion_matrix.png')
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
    result_file = os.path.join(figures_dir, f'{model_name}_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"{model_name} Model Training Results\n")
        f.write("="*80 + "\n\n")
        
        # 模型参数
        f.write("Model Parameters:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Parameters: {param_stats['total']:,}\n")
        f.write(f"Trainable Parameters: {param_stats['trainable']:,}\n")
        f.write(f"Non-trainable Parameters: {param_stats['non_trainable']:,}\n\n")
        
        # 超参数
        f.write("Hyperparameters:\n")
        f.write("-"*80 + "\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Max Length: {MAX_LENGTH}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n\n")
        
        # 训练历史
        f.write("Training History:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Epoch':<10}{'Train Loss':<15}{'Train Acc':<15}{'Val Loss':<15}{'Val Acc':<15}\n")
        f.write("-"*80 + "\n")
        for epoch in range(len(history['train_loss'])):
            f.write(f"{epoch+1:<10}{history['train_loss'][epoch]:<15.4f}{history['train_acc'][epoch]:<15.4f}")
            f.write(f"{history['val_loss'][epoch]:<15.4f}{history['val_acc'][epoch]:<15.4f}\n")
        f.write("\n")
        
        # 最佳验证结果
        f.write("="*80 + "\n")
        f.write("Best Validation Results:\n")
        f.write("-"*80 + "\n")
        f.write(f"Best Validation Accuracy: {best_val_accuracy:.4f}\n\n")
        
        # 验证集分类报告
        f.write("Validation Set - Classification Report:\n")
        f.write("-"*80 + "\n")
        f.write(val_report)
        f.write("\n\n")
        
        # 验证集混淆矩阵
        f.write("Validation Set - Confusion Matrix:\n")
        f.write("-"*80 + "\n")
        f.write(str(val_cm))
        f.write("\n\n")
        
        # 错误分布
        f.write("Error Distribution:\n")
        f.write("-"*80 + "\n")
        for name in target_names:
            stats = error_dist['class_errors'][name]
            f.write(f"{name}: {stats['wrong']}/{stats['total']} ({stats['error_rate']:.2%})\n")
        f.write("\n")
        
        # 错误样本分析
        f.write("="*80 + "\n")
        f.write("Error Samples Analysis\n")
        f.write("="*80 + "\n")
        for i, sample in enumerate(error_samples, 1):
            f.write(f"\nSample {i}:\n")
            f.write(f"Index: {sample['index']}\n")
            f.write(f"Text: {sample['text'][:200]}...\n")
            f.write(f"True Label: {target_names[sample['true_label']]}\n")
            f.write(f"Predicted Label: {target_names[sample['pred_label']]}\n")
            f.write(f"Confidence: {sample['confidence']:.4f}\n")
            f.write("-"*80 + "\n")
        f.write("\n")
        
        # 测试集结果
        f.write("="*80 + "\n")
        f.write("Test Set Results:\n")
        f.write("-"*80 + "\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Test Set - Classification Report:\n")
        f.write("-"*80 + "\n")
        f.write(test_report)
        f.write("\n\n")
        f.write("Test Set - Confusion Matrix:\n")
        f.write("-"*80 + "\n")
        f.write(str(test_cm))
        f.write("\n\n")
        
        # 推理时间
        f.write("Inference Time Statistics:\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean: {inference_stats['mean']:.4f}s\n")
        f.write(f"Std: {inference_stats['std']:.4f}s\n")
        f.write(f"Throughput: {1/inference_stats['mean']:.2f} samples/s\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")
    
    print(f"\nResults saved to: {result_file}")
    
    return {
        'history': history,
        'best_val_accuracy': best_val_accuracy,
        'test_accuracy': test_acc,
        'param_stats': param_stats,
        'inference_stats': inference_stats
    }


def main():
    """
    主函数
    Main function
    """
    # 加载数据
    tokenizer, dataset_tokenized = load_and_preprocess_data()
    
    # 定义所有可用的模型
    all_models = {
        '1': ('bert', {}),
        '2': ('bert_gru_attention', {'hidden_size': 256, 'num_layers': 1, 'dropout': 0.3}),
        '3': ('bert_transformer', {'num_heads': 8, 'num_layers': 1, 'dropout': 0.1}),
        '4': ('bert_fpn', {}),
        '5': ('transformer', {'d_model': 512, 'num_heads': 8, 'num_layers': 6, 'd_ff': 2048, 'max_len': 128, 'dropout': 0.1}),
        '6': ('gru_attention', {'embedding_dim': 300, 'hidden_size': 256, 'num_layers': 1, 'dropout': 0.3, 'bidirectional': True})
    }
    
    # 显示模型选择菜单
    print("\n" + "="*80)
    print("Available Models")
    print("="*80)
    print("1. BERT Baseline")
    print("2. BERT + GRU + Attention")
    print("3. BERT + Transformer")
    print("4. BERT + FPN")
    print("5. Pure Transformer")
    print("6. Pure GRU + Attention")
    print("7. Train All Models")
    print("="*80)
    
    # 获取用户选择
    choice = input("\nPlease select a model to train (1-7): ").strip()
    
    # 根据选择确定要训练的模型
    if choice == '7':
        models_to_train = [all_models['1'], all_models['2'], all_models['3'], all_models['4'], all_models['5'], all_models['6']]
    elif choice in all_models:
        models_to_train = [all_models[choice]]
    else:
        print("Invalid choice. Please run the script again and select a valid option (1-7).")
        return
    
    # 存储所有模型的结果
    all_results = {}
    
    # 训练和评估每个模型
    for model_name, model_kwargs in models_to_train:
        print(f"\n{'='*80}")
        print(f"Training Model: {model_name}")
        print(f"{'='*80}")
        
        # 训练（50 epoch）
        train_results = train_and_evaluate(model_name, tokenizer, dataset_tokenized, epochs=50, **model_kwargs)
        
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
