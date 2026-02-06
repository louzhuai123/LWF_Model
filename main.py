from model import Model
import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import numpy as np
import subprocess
from numpy import random
import copy

from spectral_data_loader import SpectralDataset, create_sample_spectral_data

parser = argparse.ArgumentParser(description='Continuum learning for spectral data')
parser.add_argument('--outfile', default='spectral_results.csv', type=str, help='Output file name')
parser.add_argument('--matr', default='results/spectral_acc_matr.npz', help='Accuracy matrix file name')
parser.add_argument('--num_classes', default=2, help='Number of new classes introduced each time', type=int)
parser.add_argument('--init_lr', default=0.01, type=float, help='Init learning rate')
parser.add_argument('--num_epochs', default=50, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=32, type=int, help='Mini batch size')
parser.add_argument('--input_dim', default=128, type=int, help='Spectral feature dimension')
parser.add_argument('--data_path', default='sample_spectral_data.npz', type=str, help='Path to spectral data')

# 知识蒸馏关键参数
parser.add_argument('--distill_temperature', default=2.0, type=float, help='Knowledge distillation temperature (T)')
parser.add_argument('--distill_lambda', default=1.0, type=float, help='Knowledge distillation loss weight (λ)')

args = parser.parse_args()

# 创建示例数据（如果不存在）
try:
    loaded_data = np.load(args.data_path)
    all_spectral_data = loaded_data['data']
    all_labels = loaded_data['labels']
except FileNotFoundError:
    print("创建示例光谱数据...")
    all_spectral_data, all_labels = create_sample_spectral_data(
        num_samples=2000, 
        num_features=args.input_dim, 
        num_classes=20
    )
    np.savez(args.data_path, data=all_spectral_data, labels=all_labels)

num_classes = args.num_classes
total_classes = len(np.unique(all_labels))
perm_id = np.random.permutation(total_classes)
all_classes = np.arange(total_classes)
for i in range(len(all_classes)):
    all_classes[i] = perm_id[all_classes[i]]

n_cl_temp = 0
num_iters = total_classes//num_classes
class_map = {}
map_reverse = {}
for i, cl in enumerate(all_classes):
	if cl not in class_map:
		class_map[cl] = int(n_cl_temp)
		n_cl_temp += 1

print ("Class map:", class_map)

for cl, map_cl in class_map.items():
	map_reverse[map_cl] = int(cl)

print ("Map Reverse:", map_reverse)

print ("all_classes:", all_classes)
# else:
	# perm_id = np.arange(args.total_classes)

with open(args.outfile, 'w') as file:
	print("Classes, Train Accuracy, Test Accuracy", file=file)


	#shuffle classes
	# random.shuffle(all_classes)
	# class_map = {j: int(i) for i, j in enumerate(all_classes)}
	# map_reverse = {i: int(j) for i, j in enumerate(all_classes)}
	# print('Map reverse: ', map_reverse)
	# print('Class map: ', class_map)
	# print('All classes: ', all_classes)

	model = Model(1, class_map, args)
	model.cuda()
	acc_matr = np.zeros((int(total_classes/num_classes), num_iters))
	for s in range(0, num_iters, num_classes):
		# 创建光谱数据集
		print("Loading training examples for classes", all_classes[s: s+num_classes])
		
		# 训练集：当前类别
		train_mask = np.isin(all_labels, all_classes[s:s+num_classes])
		train_spectral_data = all_spectral_data[train_mask]
		train_spectral_labels = all_labels[train_mask]
		
		train_set = SpectralDataset(
			spectral_data=train_spectral_data,
			labels=train_spectral_labels,
			classes=all_classes[s:s+num_classes],
			train=True,
			normalize=True
		)
		
		train_loader = torch.utils.data.DataLoader(
			train_set, 
			batch_size=args.batch_size,
			shuffle=True, 
			num_workers=4
		)

		# 测试集：所有已学习的类别
		test_mask = np.isin(all_labels, all_classes[:s+num_classes])
		test_spectral_data = all_spectral_data[test_mask]
		test_spectral_labels = all_labels[test_mask]
		
		test_set = SpectralDataset(
			spectral_data=test_spectral_data,
			labels=test_spectral_labels,
			classes=all_classes[:s+num_classes],
			train=False,
			normalize=True
		)
		
		test_loader = torch.utils.data.DataLoader(
			test_set, 
			batch_size=args.batch_size,
			shuffle=False, 
			num_workers=4
		)

		# Update representation via BackProp
		model.update(train_set, class_map, args)
		model.eval()

		model.n_known = model.n_classes
		print ("%d, " % model.n_known, file=file, end="")
		print ("model classes : %d, " % model.n_known)

		total = 0.0
		correct = 0.0
		for indices, spectral_features, labels in train_loader:
			spectral_features = spectral_features.cuda()
			preds = model.classify(spectral_features)
			preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
			total += labels.size(0)
			correct += (preds == labels.numpy()).sum()

		# Train Accuracy
		print ('%.2f ,' % (100.0 * correct / total), file=file, end="")
		print ('Train Accuracy : %.2f ,' % (100.0 * correct / total))

		total = 0.0
		correct = 0.0
		for indices, spectral_features, labels in test_loader:
			spectral_features = spectral_features.cuda()
			preds = model.classify(spectral_features)
			preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
			total += labels.size(0)
			correct += (preds == labels.numpy()).sum()

		# Test Accuracy
		print ('%.2f' % (100.0 * correct / total), file=file)
		print ('Test Accuracy : %.2f' % (100.0 * correct / total))

		# Accuracy matrix
		for i in range(model.n_known):
			test_set = cifar100(root='./data',
							 train=False,
							 classes=all_classes[i*num_classes: (i+1)*num_classes],
							 download=True,
							 transform=None,
							 mean_image=mean_image)
			test_loader = torch.utils.data.DataLoader(test_set, batch_size=min(500, len(test_set)),
												   shuffle=False, num_workers=12)


			total = 0.0
			correct = 0.0
			for indices, images, labels in test_loader:
				images = Variable(images).cuda()
				preds = model.classify(images)
				preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
				total += labels.size(0)
				correct += (preds == labels.numpy()).sum()
			acc_matr[i, int(s/num_classes)] = (100 * correct / total)

		print ("Accuracy matrix", acc_matr[:int(s/num_classes + 1), :int(s/num_classes + 1)])

		model.train()
		githash = subprocess.check_output(['git', 'describe', '--always'])
		np.savez(args.matr, acc_matr=acc_matr, hyper_params = args, githash=githash)
