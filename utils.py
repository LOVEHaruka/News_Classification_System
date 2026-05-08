import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict


def analyze_error_samples(dataset, model, tokenizer, device, num_samples=10):
    """
    分析错误样本
    Analyze error samples
    
    Args:
        dataset: 数据集
        model: 模型
        tokenizer: 分词器
        device: 设备
        num_samples: 显示的错误样本数量
    
    Returns:
        错误样本列表
    """
    model.eval()
    error_samples = []
    
    with torch.no_grad():
        for i in range(min(len(dataset), 1000)):  # 只分析前1000个样本
            input_ids = dataset[i]['input_ids'].unsqueeze(0).to(device)
            attention_mask = dataset[i]['attention_mask'].unsqueeze(0).to(device)
            true_label = dataset[i]['label'].item()
            
            outputs = model(input_ids, attention_mask)
            if isinstance(outputs, tuple):
                logits = outputs[1]
            else:
                logits = outputs
            
            pred_label = torch.argmax(logits, dim=1).item()
            
            if pred_label != true_label:
                # 解码文本
                text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                error_samples.append({
                    'index': i,
                    'text': text,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': torch.softmax(logits, dim=1)[0][pred_label].item()
                })
            
            if len(error_samples) >= num_samples:
                break
    
    return error_samples


def analyze_error_distribution(true_labels, pred_labels, class_names):
    """
    分析错误分布
    Analyze error distribution
    
    Args:
        true_labels: 真实标签
        pred_labels: 预测标签
        class_names: 类别名称
    
    Returns:
        错误分布统计
    """
    error_dist = {
        'class_errors': defaultdict(lambda: {'total': 0, 'correct': 0, 'wrong': 0}),
        'confusion_pairs': defaultdict(int)
    }
    
    for true, pred in zip(true_labels, pred_labels):
        class_name = class_names[true]
        error_dist['class_errors'][class_name]['total'] += 1
        
        if true == pred:
            error_dist['class_errors'][class_name]['correct'] += 1
        else:
            error_dist['class_errors'][class_name]['wrong'] += 1
            pred_class = class_names[pred]
            error_dist['confusion_pairs'][(class_name, pred_class)] += 1
    
    # 计算错误率
    for class_name in error_dist['class_errors']:
        total = error_dist['class_errors'][class_name]['total']
        wrong = error_dist['class_errors'][class_name]['wrong']
        error_dist['class_errors'][class_name]['error_rate'] = wrong / total if total > 0 else 0
    
    return error_dist


def plot_error_distribution(error_dist, class_names, output_path):
    """
    绘制错误分布图
    Plot error distribution
    
    Args:
        error_dist: 错误分布统计
        class_names: 类别名称
        output_path: 输出图片路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 各类别错误率
    classes = list(error_dist['class_errors'].keys())
    error_rates = [error_dist['class_errors'][c]['error_rate'] for c in classes]
    
    ax1.bar(classes, error_rates, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Error Rate')
    ax1.set_title('Error Rate by Class')
    ax1.set_ylim(0, 1)
    
    # 旋转x轴标签
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    
    # 添加数值标签
    for i, v in enumerate(error_rates):
        ax1.text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom')
    
    # 最常见的错误类型
    confusion_pairs = sorted(error_dist['confusion_pairs'].items(), 
                            key=lambda x: x[1], reverse=True)[:5]
    
    if confusion_pairs:
        pair_labels = [f"{p[0]} -> {p[1]}" for p, _ in confusion_pairs]
        pair_counts = [count for _, count in confusion_pairs]
        
        ax2.barh(pair_labels, pair_counts, color='lightcoral', edgecolor='darkred')
        ax2.set_xlabel('Count')
        ax2.set_ylabel('Error Type')
        ax2.set_title('Top 5 Common Errors')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Error distribution plot saved to: {output_path}")


def print_error_samples(error_samples, class_names, max_text_length=100):
    """
    打印错误样本
    Print error samples
    
    Args:
        error_samples: 错误样本列表
        class_names: 类别名称
        max_text_length: 最大文本长度
    """
    print("\n" + "="*80)
    print("Error Samples Analysis")
    print("="*80)
    
    for i, sample in enumerate(error_samples, 1):
        print(f"\nSample {i}:")
        print(f"Index: {sample['index']}")
        print(f"Text: {sample['text'][:max_text_length]}...")
        print(f"True Label: {class_names[sample['true_label']]}")
        print(f"Predicted Label: {class_names[sample['pred_label']]}")
        print(f"Confidence: {sample['confidence']:.4f}")
        print("-"*80)


def print_error_distribution(error_dist, class_names):
    """
    打印错误分布
    Print error distribution
    
    Args:
        error_dist: 错误分布统计
        class_names: 类别名称
    """
    print("\n" + "="*80)
    print("Error Distribution Analysis")
    print("="*80)
    
    print("\nError Rates by Class:")
    print("-"*80)
    for class_name in error_dist['class_errors']:
        stats = error_dist['class_errors'][class_name]
        print(f"{class_name}:")
        print(f"  Total: {stats['total']}")
        print(f"  Correct: {stats['correct']}")
        print(f"  Wrong: {stats['wrong']}")
        print(f"  Error Rate: {stats['error_rate']:.2%}")
    
    print("\nTop 10 Common Errors:")
    print("-"*80)
    confusion_pairs = sorted(error_dist['confusion_pairs'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]
    for i, ((true_class, pred_class), count) in enumerate(confusion_pairs, 1):
        print(f"{i}. {true_class} -> {pred_class}: {count} times")


def count_parameters(model):
    """
    统计模型参数量
    Count model parameters
    
    Args:
        model: 模型
    
    Returns:
        参数字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def print_parameter_stats(param_stats):
    """
    打印参数统计
    Print parameter statistics
    
    Args:
        param_stats: 参数统计信息
    """
    print("\n" + "="*80)
    print("Model Parameters")
    print("="*80)
    print(f"Total Parameters: {param_stats['total']:,}")
    print(f"Trainable Parameters: {param_stats['trainable']:,}")
    print(f"Non-trainable Parameters: {param_stats['non_trainable']:,}")
    print(f"Trainable Ratio: {param_stats['trainable']/param_stats['total']:.2%}")


def measure_inference_time(model, data_loader, device, num_batches=10):
    """
    测量推理时间
    Measure inference time
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        num_batches: 测量的批次数量
    
    Returns:
        推理时间统计
    """
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            start_time = time.time()
            _ = model(input_ids, attention_mask)
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'total': np.sum(times)
    }


def print_inference_stats(inference_stats):
    """
    打印推理统计
    Print inference statistics
    
    Args:
        inference_stats: 推理时间统计
    """
    print("\n" + "="*80)
    print("Inference Time Statistics")
    print("="*80)
    print(f"Mean: {inference_stats['mean']:.4f}s")
    print(f"Std: {inference_stats['std']:.4f}s")
    print(f"Min: {inference_stats['min']:.4f}s")
    print(f"Max: {inference_stats['max']:.4f}s")
    print(f"Total: {inference_stats['total']:.4f}s")
    print(f"Throughput: {1/inference_stats['mean']:.2f} samples/s")


def plot_training_history(history, output_path):
    """
    绘制训练历史图
    Plot training history
    
    Args:
        history: 训练历史字典，包含 train_loss, train_acc, val_loss, val_acc, test_loss, test_acc
        output_path: 输出图片路径
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss 曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s')
    if 'test_loss' in history and history['test_loss']:
        ax1.axhline(y=history['test_loss'][-1], color='g', linestyle='--', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training, Validation and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)  # 只显示整数 epoch
    
    # Accuracy 曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2, marker='o')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2, marker='s')
    if 'test_acc' in history and history['test_acc']:
        ax2.axhline(y=history['test_acc'][-1], color='g', linestyle='--', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training, Validation and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)  # 只显示整数 epoch
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to: {output_path}")


def plot_confusion_matrix(true_labels, pred_labels, class_names, output_path, title='Confusion Matrix'):
    """
    绘制混淆矩阵
    Plot confusion matrix
    
    Args:
        true_labels: 真实标签
        pred_labels: 预测标签
        class_names: 类别名称
        output_path: 输出图片路径
        title: 图表标题
    """
    cm = confusion_matrix(true_labels, pred_labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix plot saved to: {output_path}")
