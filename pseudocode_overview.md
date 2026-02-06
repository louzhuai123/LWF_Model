# 光谱数据持续学习系统 - 完整伪代码

## 1. 系统初始化

```pseudocode
ALGORITHM: SpectralContinualLearning
INPUT: 
    - spectral_dataset: 光谱特征数据 (N × feature_dim)
    - labels: 对应标签 (N,)
    - hyperparameters: {T, λ, lr, epochs, batch_size, ...}
    
OUTPUT:
    - trained_model: 能够识别所有学过类别的模型
    - accuracy_matrix: 各阶段对各类别的准确率矩阵

BEGIN
    // 1. 数据预处理
    total_classes ← UNIQUE(labels)
    class_order ← RANDOM_PERMUTATION(total_classes)
    class_map ← CREATE_CLASS_MAPPING(class_order)
    
    // 2. 模型初始化
    model ← INITIALIZE_NETWORK(input_dim=feature_dim, initial_classes=1)
    accuracy_matrix ← ZEROS(num_iterations × num_iterations)
    
    // 3. 持续学习主循环
    FOR each_iteration i = 0 to num_iterations-1:
        current_classes ← class_order[i*num_classes : (i+1)*num_classes]
        CALL INCREMENTAL_LEARNING(model, current_classes, i)
        CALL EVALUATE_ALL_TASKS(model, i)
    END FOR
END
```

## 2. 增量学习核心算法

```pseudocode
FUNCTION INCREMENTAL_LEARNING(model, new_classes, iteration):
BEGIN
    // 2.1 保存旧模型用于知识蒸馏
    IF iteration > 0:
        prev_model ← DEEP_COPY(model)
        prev_model.FREEZE_PARAMETERS()
    END IF
    
    // 2.2 扩展网络结构
    IF new_classes NOT EMPTY:
        model.INCREMENT_CLASSES(new_classes)
        model.MOVE_TO_GPU()
    END IF
    
    // 2.3 准备当前任务数据
    train_data ← FILTER_DATA_BY_CLASSES(spectral_dataset, new_classes)
    train_loader ← CREATE_DATALOADER(train_data, batch_size, shuffle=True)
    
    // 2.4 优化器设置
    optimizer ← SGD(model.parameters, lr, momentum, weight_decay)
    
    // 2.5 训练循环
    FOR epoch = 1 to num_epochs:
        FOR each batch (indices, features, labels) in train_loader:
            // 前向传播
            features ← features.TO_GPU()
            labels ← MAP_LABELS_TO_MODEL_INDICES(labels, class_map)
            logits ← model.FORWARD(features)
            
            // 计算分类损失
            cls_loss ← CROSS_ENTROPY_LOSS(logits, labels)
            
            // 计算蒸馏损失（如果不是第一个任务）
            IF iteration > 0:
                old_logits ← prev_model.FORWARD(features)
                current_old_logits ← logits[:, :prev_model.num_classes]
                distill_loss ← KNOWLEDGE_DISTILLATION_LOSS(
                    current_old_logits, old_logits, temperature=T
                )
                total_loss ← λ * distill_loss + cls_loss
            ELSE:
                total_loss ← cls_loss
            END IF
            
            // 反向传播和参数更新
            optimizer.ZERO_GRAD()
            total_loss.BACKWARD()
            optimizer.STEP()
        END FOR
    END FOR
    
    // 2.6 更新模型状态
    model.n_known ← model.n_classes
END FUNCTION
```

## 3. 知识蒸馏损失函数

```pseudocode
FUNCTION KNOWLEDGE_DISTILLATION_LOSS(student_logits, teacher_logits, T):
BEGIN
    // 3.1 温度缩放的softmax
    student_soft ← SOFTMAX(student_logits / T)
    teacher_soft ← SOFTMAX(teacher_logits / T)
    
    // 3.2 计算KL散度损失
    log_student ← LOG_SOFTMAX(student_logits / T)
    kd_loss ← -MEAN(SUM(teacher_soft * log_student, dim=1))
    
    RETURN kd_loss
END FUNCTION
```

## 4. 网络结构动态扩展

```pseudocode
FUNCTION INCREMENT_CLASSES(model, new_classes):
BEGIN
    n ← LENGTH(new_classes)
    
    // 4.1 获取当前分类层参数
    current_weights ← model.fc.weight.DATA
    in_features ← model.fc.in_features
    out_features ← model.fc.out_features
    
    // 4.2 创建新的分类层
    new_out_features ← out_features + n
    new_fc ← LINEAR_LAYER(in_features, new_out_features, bias=False)
    
    // 4.3 权重初始化和复制
    KAIMING_NORMAL_INIT(new_fc.weight)
    new_fc.weight.DATA[:out_features] ← current_weights
    
    // 4.4 替换模型的分类层
    model.fc ← new_fc
    model.n_classes ← model.n_classes + n
END FUNCTION
```

## 5. 全面评估算法

```pseudocode
FUNCTION EVALUATE_ALL_TASKS(model, current_iteration):
BEGIN
    model.EVAL_MODE()
    
    // 5.1 评估当前任务性能
    current_classes ← class_order[0 : (current_iteration+1)*num_classes]
    
    // 训练集准确率
    train_accuracy ← COMPUTE_ACCURACY(model, current_classes, train=True)
    PRINT("Train Accuracy:", train_accuracy)
    
    // 测试集准确率  
    test_accuracy ← COMPUTE_ACCURACY(model, current_classes, train=False)
    PRINT("Test Accuracy:", test_accuracy)
    
    // 5.2 计算准确率矩阵（每个任务对每个任务的性能）
    FOR task_id = 0 to current_iteration:
        task_classes ← class_order[task_id*num_classes : (task_id+1)*num_classes]
        task_accuracy ← COMPUTE_ACCURACY(model, task_classes, train=False)
        accuracy_matrix[task_id, current_iteration] ← task_accuracy
    END FOR
    
    PRINT("Accuracy Matrix:", accuracy_matrix[:current_iteration+1, :current_iteration+1])
    model.TRAIN_MODE()
END FUNCTION
```

## 6. 准确率计算辅助函数

```pseudocode
FUNCTION COMPUTE_ACCURACY(model, target_classes, train):
BEGIN
    // 6.1 准备数据
    IF train:
        data ← FILTER_DATA_BY_CLASSES(train_spectral_dataset, target_classes)
    ELSE:
        data ← FILTER_DATA_BY_CLASSES(test_spectral_dataset, target_classes)
    END IF
    
    data_loader ← CREATE_DATALOADER(data, batch_size, shuffle=False)
    
    // 6.2 预测和统计
    total_samples ← 0
    correct_predictions ← 0
    
    WITH NO_GRADIENT():
        FOR each batch (indices, features, true_labels) in data_loader:
            features ← features.TO_GPU()
            predictions ← model.CLASSIFY(features)
            predictions ← MAP_PREDICTIONS_TO_ORIGINAL_LABELS(predictions, class_map)
            
            total_samples ← total_samples + LENGTH(true_labels)
            correct_predictions ← correct_predictions + SUM(predictions == true_labels)
        END FOR
    END WITH
    
    accuracy ← 100.0 * correct_predictions / total_samples
    RETURN accuracy
END FUNCTION
```

## 7. 数据处理流水线

```pseudocode
FUNCTION SPECTRAL_DATA_PREPROCESSING(raw_spectral_data, labels):
BEGIN
    // 7.1 数据标准化
    scaler ← STANDARD_SCALER()
    normalized_data ← scaler.FIT_TRANSFORM(raw_spectral_data)
    
    // 7.2 数据类型转换
    tensor_data ← CONVERT_TO_TENSOR(normalized_data, dtype=FLOAT32)
    tensor_labels ← CONVERT_TO_TENSOR(labels, dtype=INT64)
    
    RETURN tensor_data, tensor_labels
END FUNCTION

FUNCTION CREATE_DATALOADER(spectral_data, labels, classes, batch_size, shuffle):
BEGIN
    // 7.3 类别过滤
    mask ← CREATE_MASK_FOR_CLASSES(labels, classes)
    filtered_data ← spectral_data[mask]
    filtered_labels ← labels[mask]
    
    // 7.4 数据集封装
    dataset ← SPECTRAL_DATASET(filtered_data, filtered_labels)
    dataloader ← DATALOADER(dataset, batch_size, shuffle, num_workers=4)
    
    RETURN dataloader
END FUNCTION
```

## 8. 网络架构定义

```pseudocode
FUNCTION INITIALIZE_NETWORK(input_dim, initial_classes):
BEGIN
    // 8.1 特征提取器（全连接网络）
    feature_extractor ← SEQUENTIAL(
        LINEAR(input_dim, 512),
        BATCH_NORM_1D(512),
        RELU(),
        DROPOUT(0.3),
        
        LINEAR(512, 256),
        BATCH_NORM_1D(256),
        RELU(),
        DROPOUT(0.3),
        
        LINEAR(256, 128),
        BATCH_NORM_1D(128),
        RELU(),
        DROPOUT(0.2)
    )
    
    // 8.2 分类器
    classifier ← LINEAR(128, initial_classes, bias=False)
    
    // 8.3 权重初始化
    APPLY_KAIMING_NORMAL_INIT(feature_extractor)
    APPLY_KAIMING_NORMAL_INIT(classifier)
    
    // 8.4 模型组装
    model ← CREATE_MODEL(feature_extractor, classifier)
    model.n_classes ← initial_classes
    model.n_known ← 0
    
    RETURN model
END FUNCTION
```

## 9. 主程序执行流程

```pseudocode
MAIN_PROGRAM:
BEGIN
    // 9.1 参数解析
    args ← PARSE_COMMAND_LINE_ARGUMENTS()
    
    // 9.2 数据加载
    spectral_data, labels ← LOAD_SPECTRAL_DATA(args.data_path)
    
    // 9.3 执行持续学习
    model, accuracy_matrix ← SpectralContinualLearning(
        spectral_data, labels, args
    )
    
    // 9.4 结果保存
    SAVE_RESULTS(accuracy_matrix, args.output_file)
    SAVE_MODEL(model, args.model_file)
    
    // 9.5 性能分析
    ANALYZE_FORGETTING(accuracy_matrix)
    PLOT_LEARNING_CURVE(accuracy_matrix)
END
```

## 关键特性总结

1. **增量学习**: 逐步学习新类别，无需重新训练整个模型
2. **知识蒸馏**: 使用温度缩放的softmax保持旧知识
3. **动态网络**: 根据新任务动态扩展分类层
4. **防遗忘**: 通过λ平衡新旧任务的学习
5. **光谱适配**: 专门针对一维光谱特征设计的网络结构

## 参数调优策略

- **T (温度)**: 2.0-4.0，控制知识传递的"软度"
- **λ (权重)**: 0.5-2.0，平衡新旧任务重要性
- **学习率**: 0.001-0.01，根据数据规模调整
- **批次大小**: 16-64，根据内存和数据量调整