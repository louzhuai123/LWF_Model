#!/usr/bin/env python3
"""
测试光谱数据的持续学习模型
"""

import torch
import numpy as np
from spectral_data_loader import SpectralDataset, create_sample_spectral_data
from model import Model
import argparse

def test_spectral_model():
    """测试光谱模型的基本功能"""
    
    # 创建测试参数
    class Args:
        def __init__(self):
            self.input_dim = 64  # 光谱特征维度
            self.init_lr = 0.01
            self.num_epochs = 5
            self.batch_size = 16
            # 知识蒸馏参数
            self.distill_temperature = 3.0  # 蒸馏温度
            self.distill_lambda = 0.5       # 蒸馏损失权重
    
    args = Args()
    
    # 创建示例光谱数据
    print("创建示例光谱数据...")
    spectral_data, labels = create_sample_spectral_data(
        num_samples=200, 
        num_features=args.input_dim, 
        num_classes=6
    )
    
    print(f"数据形状: {spectral_data.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"类别数量: {len(np.unique(labels))}")
    
    # 创建类别映射
    all_classes = np.unique(labels)
    class_map = {cl: i for i, cl in enumerate(all_classes)}
    
    # 创建模型
    print("创建模型...")
    model = Model(1, class_map, args)  # 初始1个类别
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("使用GPU")
    else:
        print("使用CPU")
    
    # 测试第一批类别 (类别0, 1)
    print("\n=== 测试第一批类别 (0, 1) ===")
    first_classes = [0, 1]
    
    # 创建数据集
    train_mask = np.isin(labels, first_classes)
    train_data = spectral_data[train_mask]
    train_labels = labels[train_mask]
    
    dataset = SpectralDataset(
        spectral_data=train_data,
        labels=train_labels,
        classes=first_classes,
        train=True,
        normalize=True
    )
    
    print(f"第一批训练样本数: {len(dataset)}")
    
    # 测试前向传播
    sample_batch = torch.randn(4, args.input_dim)
    if torch.cuda.is_available():
        sample_batch = sample_batch.cuda()
    
    with torch.no_grad():
        output = model(sample_batch)
        print(f"模型输出形状: {output.shape}")
        
        # 测试分类
        preds = model.classify(sample_batch)
        print(f"预测结果: {preds}")
    
    # 模拟训练更新
    print("模拟训练更新...")
    model.update(dataset, class_map, args)
    
    # 测试第二批类别 (类别2, 3)
    print("\n=== 测试第二批类别 (2, 3) ===")
    second_classes = [2, 3]
    
    # 创建第二批数据集
    train_mask = np.isin(labels, second_classes)
    train_data = spectral_data[train_mask]
    train_labels = labels[train_mask]
    
    dataset2 = SpectralDataset(
        spectral_data=train_data,
        labels=train_labels,
        classes=second_classes,
        train=True,
        normalize=True
    )
    
    print(f"第二批训练样本数: {len(dataset2)}")
    
    # 更新模型
    model.update(dataset2, class_map, args)
    
    # 测试最终模型
    print(f"\n最终模型类别数: {model.n_classes}")
    
    # 测试所有类别的预测
    test_mask = np.isin(labels, [0, 1, 2, 3])
    test_data = spectral_data[test_mask]
    test_labels = labels[test_mask]
    
    test_dataset = SpectralDataset(
        spectral_data=test_data,
        labels=test_labels,
        classes=[0, 1, 2, 3],
        train=False,
        normalize=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=16, 
        shuffle=False
    )
    
    # 计算准确率
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for indices, features, target_labels in test_loader:
            if torch.cuda.is_available():
                features = features.cuda()
            
            preds = model.classify(features)
            preds = preds.cpu().numpy()
            
            total += len(target_labels)
            correct += (preds == target_labels.numpy()).sum()
    
    accuracy = 100.0 * correct / total
    print(f"测试准确率: {accuracy:.2f}%")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_spectral_model()