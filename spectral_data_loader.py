import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SpectralDataset(Dataset):
    def __init__(self, data_path=None, 
                 classes=None,
                 train=True,
                 transform=None,
                 target_transform=None,
                 spectral_data=None,
                 labels=None,
                 normalize=True):
        """
        光谱数据集加载器
        
        Args:
            data_path: 数据文件路径 (CSV, NPY等)
            classes: 要包含的类别列表
            train: 是否为训练集
            spectral_data: 直接传入的光谱数据 (N, feature_dim)
            labels: 直接传入的标签 (N,)
            normalize: 是否标准化特征
        """
        
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = normalize
        
        # 加载数据
        if spectral_data is not None and labels is not None:
            self.data = spectral_data
            self.labels = labels
        elif data_path is not None:
            self.data, self.labels = self._load_data(data_path)
        else:
            raise ValueError("必须提供 data_path 或者 (spectral_data, labels)")
        
        # 筛选指定类别
        if classes is not None:
            self._filter_classes(classes)
        
        # 数据预处理
        if self.normalize:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
        
        # 转换为tensor
        self.data = torch.FloatTensor(self.data)
        self.labels = np.array(self.labels)
        
    def _load_data(self, data_path):
        """从文件加载数据"""
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            # 假设最后一列是标签，其余是特征
            data = df.iloc[:, :-1].values
            labels = df.iloc[:, -1].values
        elif data_path.endswith('.npy'):
            # 假设是包含数据和标签的numpy数组
            loaded = np.load(data_path, allow_pickle=True)
            if isinstance(loaded, dict):
                data = loaded['data']
                labels = loaded['labels']
            else:
                # 假设最后一列是标签
                data = loaded[:, :-1]
                labels = loaded[:, -1]
        else:
            raise ValueError(f"不支持的文件格式: {data_path}")
        
        return data.astype(np.float32), labels.astype(np.int64)
    
    def _filter_classes(self, classes):
        """筛选指定类别的数据"""
        mask = np.isin(self.labels, classes)
        self.data = self.data[mask]
        self.labels = self.labels[mask]
    
    def __getitem__(self, index):
        """获取单个样本"""
        spectral_features = self.data[index]
        label = self.labels[index]
        
        if self.transform:
            spectral_features = self.transform(spectral_features)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return index, spectral_features, label
    
    def __len__(self):
        return len(self.data)

# 示例使用函数
def create_sample_spectral_data(num_samples=1000, num_features=128, num_classes=10):
    """
    创建示例光谱数据用于测试
    """
    np.random.seed(42)
    
    # 生成模拟光谱特征
    data = []
    labels = []
    
    for class_id in range(num_classes):
        # 每个类别生成特定的光谱模式
        class_samples = num_samples // num_classes
        
        # 基础光谱模式 + 噪声
        base_pattern = np.random.randn(num_features) * 0.5 + class_id * 0.3
        
        for _ in range(class_samples):
            # 添加样本级别的变化
            sample = base_pattern + np.random.randn(num_features) * 0.1
            data.append(sample)
            labels.append(class_id)
    
    return np.array(data), np.array(labels)

if __name__ == "__main__":
    # 创建示例数据
    data, labels = create_sample_spectral_data()
    
    # 保存示例数据
    np.savez('sample_spectral_data.npz', data=data, labels=labels)
    
    # 测试数据加载器
    dataset = SpectralDataset(
        spectral_data=data[:800],  # 训练集
        labels=labels[:800],
        classes=[0, 1, 2],  # 只使用前3个类别
        train=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"特征维度: {dataset.data.shape[1]}")
    
    # 测试单个样本
    idx, features, label = dataset[0]
    print(f"样本形状: {features.shape}")
    print(f"标签: {label}")