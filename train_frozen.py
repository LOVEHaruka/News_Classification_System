import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import Dataset
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

from models.bert_frozen import BertBaselineFrozen
from utils import (
    plot_training_history, plot_confusion_matrix,
    analyze_error_distribution, plot_error_distribution,
    analyze_error_samples, print_error_samples,
    print_error_distribution, print_inference_stats,
    measure_inference_time, count_parameters
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
NUM_CLASSES = 4


def load_data():
    """
    加载数据
    Load data
    """
    print(f"\n{'='*80}")
    print("Loading AG News dataset from local files...")
    print(f"{'='*80}")
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    return train_df, val_df, test_df


def tokenize_data(train_df, val_df, test_df, tokenizer, max_length):
    """
    分词数据
    Tokenize data
    """
    print(f"\n{'='*80}")
    print("Tokenizing dataset...")
    print(f"{'='*80}")
    
    # 合并 Title 和 Description 为 text，处理标签
    def process_df(df):
        # 合并 Title 和 Description
        df['text'] = df['Title'] + ' ' + df['Description']
        # 转换 Class Index 为 0-3（原始是 1-4）
        df['label'] = df['Class Index'] - 1
        return df[['text', 'label']]
    
    train_df = process_df(train_df)
    val_df = process_df(val_df)
    test_df = process_df(test_df)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    train_dataset = train_dataset.remove_columns(['text'])
    val_dataset = val_dataset.remove_columns(['text'])
    test_dataset = test_dataset.remove_columns(['text'])
    
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    print("Tokenization completed!")
    
    return train_dataset, val_dataset, test_dataset


def train_epoch(model, dataloader, optimizer, device):
    """
    训练一个 epoch
    Train for one epoch
    """
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
        
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """
    评估模型
    Evaluate model
    """
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


def train_and_evaluate(freeze_layers=8, epochs=20):
    """
    训练和评估模型
    Train and evaluate model
    """
    BATCH_SIZE = 16
    MAX_LENGTH = 64
    LEARNING_RATE = 3e-5
    EARLY_STOP_PATIENCE = 3  # 早停耐心值
    
    print(f"\n{'='*80}")
    print("BERT Frozen Fine-tuning")
    print(f"{'='*80}")
    print(f"Freeze Layers: {freeze_layers}")
    print(f"Training Top Layers: {12 - freeze_layers}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Length: {MAX_LENGTH}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Early Stop Patience: {EARLY_STOP_PATIENCE}")
    print(f"{'='*80}")
    
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_df, val_df, test_df = load_data()
    train_dataset, val_dataset, test_dataset = tokenize_data(
        train_df, val_df, test_df, tokenizer, MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    model = BertBaselineFrozen(
        num_labels=NUM_CLASSES,
        model_name='bert-base-uncased',
        freeze_layers=freeze_layers
    ).to(device)
    
    total_params, trainable_params = model.get_num_parameters()
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters: {total_params - trainable_params:,}")
    print(f"Trainable Ratio: {trainable_params / total_params * 100:.2f}%")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_accuracy = 0
    best_model_dir = os.path.join(BASE_DIR, f'bert_frozen_{freeze_layers}_model')
    os.makedirs(best_model_dir, exist_ok=True)
    
    # 早停计数器
    patience_counter = 0
    
    print(f"\n{'='*80}")
    print("Starting Training...")
    print(f"{'='*80}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 80)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
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
    
    model.load_state_dict(torch.load(os.path.join(best_model_dir, 'model.pth')))
    
    test_loss, test_acc, test_labels, test_preds = evaluate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    history['test_loss'] = [test_loss] * len(history['train_loss'])
    history['test_acc'] = [test_acc] * len(history['train_acc'])
    
    figures_dir = os.path.join(BASE_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    history_path = os.path.join(figures_dir, f'bert_frozen_{freeze_layers}_training_history.png')
    plot_training_history(history, history_path)
    
    target_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    
    print(f"\n{'='*80}")
    print("Test Set - Classification Report:")
    print(f"{'='*80}")
    test_report = classification_report(test_labels, test_preds, target_names=target_names)
    print(test_report)
    
    print(f"\n{'='*80}")
    print("Test Set - Confusion Matrix:")
    print(f"{'='*80}")
    test_cm = confusion_matrix(test_labels, test_preds)
    print(test_cm)
    
    test_cm_path = os.path.join(figures_dir, f'bert_frozen_{freeze_layers}_test_confusion_matrix.png')
    plot_confusion_matrix(
        test_labels, test_preds, target_names, test_cm_path,
        title=f'BERT Frozen {freeze_layers} - Test Set Confusion Matrix'
    )
    
    inference_stats = measure_inference_time(model, test_loader, device, num_batches=10)
    print_inference_stats(inference_stats)
    
    result_file = os.path.join(figures_dir, f'bert_frozen_{freeze_layers}_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"BERT Frozen {freeze_layers} Layers Model Results\n")
        f.write("="*80 + "\n\n")
        
        f.write("Model Parameters:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Frozen Parameters: {total_params - trainable_params:,}\n")
        f.write(f"Trainable Ratio: {trainable_params / total_params * 100:.2f}%\n\n")
        
        f.write("Hyperparameters:\n")
        f.write("-"*80 + "\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Max Length: {MAX_LENGTH}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Freeze Layers: {freeze_layers}\n\n")
        
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
        f.write(f"Average Inference Time: {inference_stats['mean']*1000:.4f} ms\n")
        f.write(f"Samples per Second: {1/inference_stats['mean']*BATCH_SIZE:.2f}\n")
    
    print(f"\nResults saved to: {result_file}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'history': history
    }


def main():
    """
    主函数
    Main function
    """
    print("\n" + "="*80)
    print("BERT Frozen Fine-tuning")
    print("="*80)
    print("Select the number of frozen layers:")
    print("1. Freeze 8 layers (train top 4 layers)")
    print("2. Freeze 10 layers (train top 2 layers)")
    print("3. Freeze 11 layers (train top 1 layer)")
    print("4. Test all configurations")
    print("="*80)
    
    choice = input("\nPlease select an option (1-4): ").strip()
    
    freeze_options = {
        '1': 8,
        '2': 10,
        '3': 11
    }
    
    if choice == '4':
        all_results = {}
        for name, freeze_layers in freeze_options.items():
            print(f"\n{'='*80}")
            print(f"Testing configuration: Freeze {freeze_layers} layers")
            print(f"{'='*80}")
            results = train_and_evaluate(freeze_layers=freeze_layers, epochs=20)
            all_results[name] = results
        
        print(f"\n{'='*80}")
        print("Comparison of Different Freeze Configurations")
        print(f"{'='*80}")
        print(f"{'Frozen Layers':<20}{'Trainable Params':<25}{'Test Accuracy':<20}")
        print("-"*80)
        for name, freeze_layers in freeze_options.items():
            results = all_results[name]
            trainable = results['trainable_params']
            acc = results['test_accuracy']
            print(f"{freeze_layers:<20}{trainable:<25,}{acc:<20.4f}")
    
    elif choice in freeze_options:
        freeze_layers = freeze_options[choice]
        train_and_evaluate(freeze_layers=freeze_layers, epochs=20)
    
    else:
        print("Invalid choice. Please run the script again and select a valid option (1-4).")


if __name__ == '__main__':
    main()
