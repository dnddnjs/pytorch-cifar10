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
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', default=128, help='')
parser.add_argument('--num_worker', default=4, help='')
parser.add_argument('--valid_size', default=0.1, help='')
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

# split train dataset into train and valid. After that, make sampler for each dataset
# code from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
num_train = len(dataset_train)
indices = list(range(num_train))
split = int(np.floor(args.valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, 
	                      num_workers=args.num_worker, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, 
	                      num_workers=args.num_worker, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=100, 
	                     shuffle=False, num_workers=args.num_worker)

# there are 10 classes so the dataset name is cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
	       'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Making model..')
controller_model = Controller().to(device)
criterion = nn.CrossEntropyLoss()
controller_optimizer = optim.Adam(controller_model.parameters(), lr=0.0035)


def train_child(epoch, model, child_optimizer, normal_arc, reduction_arc):
	model.to(device)
	model.train()

	train_loss = 0
	correct = 0
	total = 0

	# cosine_lr_scheduler.step()

	for batch_idx, (inputs, targets) in enumerate(train_loader):
		inputs = inputs.to(device)
		targets = targets.to(device)
		outputs = model(inputs, normal_arc, reduction_arc)
		loss = criterion(outputs, targets)

		child_optimizer.zero_grad()
		loss.backward()
		child_optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		if batch_idx % 10 == 0:
			print('train child epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(epoch, batch_idx, 
			 	len(train_loader), train_loss/(batch_idx+1), 100.*correct/total))

	model.eval()
	test_loss = 0
	correct = 0
	total = 0
	'''
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(valid_loader):
			inputs = inputs.to(device)
			targets = targets.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			
			if batch_idx % 10 == 0:
				print('epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(epoch, batch_idx, 
			  	len(valid_loader), test_loss/(batch_idx+1), 100 * correct/total))
	'''
	return model


def train_controller(child, controller, running_reward, entropy_seq, log_prob_seq, normal_arc, reduction_arc):
	child.eval()
	controller.train()

	# todo: in paper controller has to be updated for 2000 times.
	# but there is no information about mini-batch size. Need to be checked
	# question: if update controller once then the child is the output of previous controller model
	# is it right to update controller after gather all reward from validation dataset?
	for batch_idx, (inputs, targets) in enumerate(valid_loader):
		correct = 0
		total = 0

		# 1. get reward using a single mini-batch of validation data
		inputs = inputs.to(device)
		targets = targets.to(device)
		outputs = child(inputs, normal_arc, reduction_arc)
	
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		reward = correct / total

		# 2. using the reward, train controller with REINFORCE
		running_reward = 0.99 * running_reward + 0.01 * reward
		baseline = running_reward
		log_prob = torch.cat(log_prob_seq, dim=0).sum()
		entropy = torch.cat(entropy_seq, dim=0).sum()
		entropy_bonus = entropy

		loss = - log_prob * (reward - baseline)
		loss = loss - 0.0001 * entropy_bonus

		controller_optimizer.zero_grad()
		loss.backward(retain_graph=True)
		controller_optimizer.step()
		
		if batch_idx % 10 == 0:
			print('train controller : [{}/{}]| loss: {:.3f} | reward: {:.3f}'.format(batch_idx, 
				len(valid_loader), loss.item(), reward))

	return controller, running_reward


def test_final_model(model):
	model.eval()

	test_loss = 0
	correct = 0
	total = 0

	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(test_loader):
			inputs = inputs.to(device)
			targets = targets.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			
			print('epoch : [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(epoch, batch_idx, 
			  len(test_loader), test_loss/(batch_idx+1), 100 * correct/total))

	acc = 100 * correct / total
	return acc


def main(controller_model, cosine_annealing_scheduler):
	running_reward = 0
	for epoch in range(310):
		outputs = controller_model.sample_child()
		normal_arc, reduction_arc, entropy_seq, log_prob_seq = outputs
		model = Child()

		if epoch == 0:
			child_optimizer = optim.SGD(model.parameters(), lr=0.05, 
		                      momentum=0.9, weight_decay=1e-4, nesterov=True)
		else:
			model.load_state_dict(torch.load('./save_model/child.pt'))

		# cosine_lr_scheduler = cosine_annealing_scheduler(child_optimizer, lr=0.05)
		model = train_child(epoch, model, child_optimizer, normal_arc, reduction_arc)
		torch.save(model.state_dict(), './save_model/child.pt')

		if epoch < 310 - 1:
			controller_model, running_reward = train_controller(model, controller_model, running_reward, entropy_seq, log_prob_seq, normal_arc, reduction_arc)

	final_model = model
	final_acc = test_final_model(final_model)
	print('training is done. Final accuarcy is ', final_acc)


if __name__=="__main__":
	main(controller_model, cosine_annealing_scheduler)