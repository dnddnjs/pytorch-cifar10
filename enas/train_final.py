import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from controller import Controller
from child import Child
from cosine_optim import cosine_annealing_scheduler

import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--batch_size', default=144, help='')
parser.add_argument('--num_worker', default=16, help='')
parser.add_argument('--valid_size', default=0.1, help='')
parser.add_argument('--epochs', default=630, help='')
parser.add_argument('--dropout', default=0.8, help='dropout rate')
parser.add_argument('--use_auxiliary', default=False, action='store_true', help='auxiliary loss for child.')
parser.add_argument('--use_drop_path', default=False, action='store_true', help='drop path for child.')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')
transforms_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transforms_valid = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transforms_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset_train = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms_train)
dataset_valid = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms_valid)
dataset_test = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms_test)

num_train = len(dataset_train)
indices = list(range(num_train))
split = int(np.floor(args.valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
valid_sampler = SubsetRandomSampler(valid_idx)

valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, 
										  num_workers=args.num_worker, sampler=valid_sampler, pin_memory=True)

controller = Controller().to(device)
child = Child(dropout_rate=float(args.dropout), use_auxiliary=args.use_auxiliary).to(device)
controller.load_state_dict(torch.load('./save_model/controller.pth'))
child.load_state_dict(torch.load('./save_model/child.pth'))

print('-'*40)
print('1. get 10 architecture and compute the reward of each architecture. Take the architecture with highest reward')
controller.eval()
child.eval()

best_reward = 0
for i in range(10):
	controller.init_hidden(batch_size=1)
	outputs = controller.sample_child()
	normal_arc, reduction_arc, _, _ = outputs

	for batch_idx, (inputs, targets) in enumerate(valid_loader):
		outputs = controller.sample_child()
		normal_arc, reduction_arc, entropy_seq, log_prob_seq = outputs
		correct = 0
		total = 0

		# 1. get reward using a single mini-batch of validation data
		inputs = inputs.to(device)
		targets = targets.to(device)
		outputs, aux_outs = child(inputs, normal_arc, reduction_arc)
	
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		reward = correct / total
		
		if reward >= best_reward:
			best_reward = reward
			best_reward_arc = [normal_arc, reduction_arc]
		print('controller : [{}/10]| reward: {:.3f} '.format(i+1, reward))
		break

print('best reward is ', best_reward)
print('best architecture is ')# , best_reward_arc)
print('\t normal arc    :', [na.item() for na in best_reward_arc[0]])
print('\t reduction arc :', [na.item() for na in best_reward_arc[1]])

print('-'*40)
print('2. train child model from scratch with best reward architecture')
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, 
	                      shuffle=True, num_workers=args.num_worker)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=100, 
	                     shuffle=False, num_workers=args.num_worker)

# there are 10 classes so the dataset name is cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
	       'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Making model..')

criterion = nn.CrossEntropyLoss()
normal_arc = best_reward_arc[0]
reduction_arc = best_reward_arc[1]

child = Child(dropout_rate=float(args.dropout), use_auxiliary=args.use_auxiliary).to(device)
child_optimizer = optim.SGD(child.parameters(), lr=0.05, 
		                      momentum=0.9, weight_decay=2e-4, nesterov=True)
cosine_lr_scheduler = cosine_annealing_scheduler(child_optimizer, lr_max=0.05, lr_min=0.0001)

best_acc = 0
for epoch in range(args.epochs):
	child.train()
	cosine_lr_scheduler.step()
	
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		inputs = inputs.to(device)
		targets = targets.to(device)
		outputs, aux_outs = child(inputs, normal_arc, reduction_arc)
		loss = criterion(outputs, targets)
		if args.use_auxiliary:
			loss += 0.4 * criterion(aux_outs, targets)

		child_optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(child.parameters(), 5.0)
		child_optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		if batch_idx % 10 == 0:
			print('train child epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(epoch, batch_idx, 
				len(train_loader), train_loss/(batch_idx+1), 100.*correct/total))

	child.eval()

	test_loss = 0
	correct = 0
	total = 0

	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(test_loader):
			inputs = inputs.to(device)
			targets = targets.to(device)
			outputs, aux_outs = child(inputs, normal_arc, reduction_arc)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

	acc = 100 * correct / total
	print('epoch : [{}/310]| acc: {:.3f}'.format(epoch, acc))

	if acc >= best_acc:
		print('best accuracy is ', best_acc)
		best_acc = acc
		torch.save(child.state_dict(), './save_model/best_child.pth')

