import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import copy

import torchvision.models as models
import torchvision.transforms as transforms

def MultiClassCrossEntropy(logits, labels, T):
	# Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
	labels = Variable(labels.data, requires_grad=False).cuda()
	outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
	labels = torch.softmax(labels/T, dim=1)
	# print('outputs: ', outputs)
	# print('labels: ', labels.shape)
	outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
	outputs = -torch.mean(outputs, dim=0, keepdim=False)
	# print('OUT: ', outputs)
	return Variable(outputs.data, requires_grad=True).cuda()

def kaiming_normal_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

class Model(nn.Module):
	def __init__(self, classes, classes_map, args):
		# Hyper Parameters
		self.init_lr = args.init_lr
		self.num_epochs = args.num_epochs
		self.batch_size = args.batch_size
		self.lower_rate_epoch = [int(0.7 * self.num_epochs), int(0.9 * self.num_epochs)] #hardcoded decay schedule
		self.lr_dec_factor = 10
		
		# 知识蒸馏关键参数
		self.distill_temperature = getattr(args, 'distill_temperature', 2.0)  # 蒸馏温度 T
		self.distill_lambda = getattr(args, 'distill_lambda', 1.0)  # 蒸馏损失权重 λ
		
		self.pretrained = False
		self.momentum = 0.9
		self.weight_decay = 0.0001
		# Constant to provide numerical stability while normalizing
		self.epsilon = 1e-16

		# Network architecture for spectral features
		super(Model, self).__init__()
		
		# 假设光谱特征向量维度，可以通过args传入
		self.input_dim = getattr(args, 'input_dim', 128)  # 默认128维光谱特征
		
		# 构建全连接网络替代ResNet
		self.feature_extractor = nn.Sequential(
			nn.Linear(self.input_dim, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Dropout(0.3),
			
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(inplace=True),
			nn.Dropout(0.3),
			
			nn.Linear(256, 128),
			nn.BatchNorm1d(128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.2)
		)
		
		# 分类层
		self.fc = nn.Linear(128, classes, bias=False)
		
		# 应用权重初始化
		self.feature_extractor.apply(kaiming_normal_init)
		kaiming_normal_init(self.fc) 


		# n_classes is incremented before processing new data in an iteration
		# n_known is set to n_classes after all data for an iteration has been processed
		self.n_classes = 0
		self.n_known = 0
		self.classes_map = classes_map

	def forward(self, x):
		# x shape: (batch_size, input_dim) - 一维光谱特征
		x = self.feature_extractor(x)  # 输出: (batch_size, 128)
		x = self.fc(x)  # 输出: (batch_size, num_classes)
		return x

	def increment_classes(self, new_classes):
		"""Add n classes in the final fc layer"""
		n = len(new_classes)
		print('new classes: ', n)
		in_features = self.fc.in_features  # 现在是128
		out_features = self.fc.out_features
		weight = self.fc.weight.data

		if self.n_known == 0:
			new_out_features = n
		else:
			new_out_features = out_features + n
		print('new out features: ', new_out_features)
		
		# 重新创建分类层
		self.fc = nn.Linear(in_features, new_out_features, bias=False)
		
		kaiming_normal_init(self.fc.weight)
		self.fc.weight.data[:out_features] = weight
		self.n_classes += n

	def classify(self, images):
		"""Classify images by softmax

		Args:
			x: input image batch
		Returns:
			preds: Tensor of size (batch_size,)
		"""
		_, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)

		return preds

	def update(self, dataset, class_map, args):

		self.compute_means = True

		# Save a copy to compute distillation outputs
		prev_model = copy.deepcopy(self)
		prev_model.cuda()

		classes = list(set(dataset.train_labels))
		#print("Classes: ", classes)
		print('Known: ', self.n_known)
		if self.n_classes == 1 and self.n_known == 0:
			new_classes = [classes[i] for i in range(1,len(classes))]
		else:
			new_classes = [cl for cl in classes if class_map[cl] >= self.n_known]

		if len(new_classes) > 0:
			self.increment_classes(new_classes)
			self.cuda()

		loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
											   shuffle=True, num_workers=12)

		print("Batch Size (for n_classes classes) : ", len(dataset))
		optimizer = optim.SGD(self.parameters(), lr=self.init_lr, momentum = self.momentum, weight_decay=self.weight_decay)

		with tqdm(total=self.num_epochs) as pbar:
			for epoch in range(self.num_epochs):
				
				# Modify learning rate
				# if (epoch+1) in lower_rate_epoch:
				# 	self.lr = self.lr * 1.0/lr_dec_factor
				# 	for param_group in optimizer.param_groups:
				# 		param_group['lr'] = self.lr

				
				for i, (indices, spectral_features, labels) in enumerate(loader):
					seen_labels = []
					spectral_features = spectral_features.cuda()
					seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
					labels = seen_labels.cuda()

					optimizer.zero_grad()
					logits = self.forward(spectral_features)
					cls_loss = nn.CrossEntropyLoss()(logits, labels)
					
					if self.n_classes//len(new_classes) > 1:
						# 知识蒸馏损失
						dist_target = prev_model.forward(spectral_features)
						logits_dist = logits[:,:-(self.n_classes-self.n_known)]
						dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, self.distill_temperature)
						
						# 总损失 = λ * 蒸馏损失 + 分类损失
						loss = self.distill_lambda * dist_loss + cls_loss
						
						# 打印损失信息（可选）
						if (i+1) % 10 == 0:
							tqdm.write(f'Distill Loss: {dist_loss.item():.4f}, Cls Loss: {cls_loss.item():.4f}, λ: {self.distill_lambda}, T: {self.distill_temperature}')
					else:
						loss = cls_loss




					loss.backward()
					optimizer.step()

					if (i+1) % 1 == 0:
						tqdm.write('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
							   %(epoch+1, self.num_epochs, i+1, np.ceil(len(dataset)/self.batch_size), loss.data))

				pbar.update(1)






