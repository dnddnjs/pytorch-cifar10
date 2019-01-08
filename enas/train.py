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
parser.add_argument('--resume', default=None, help='path of the model weight.')
parser.add_argument('--batch_size', default=160, help='')
parser.add_argument('--num_worker', default=4, help='')
parser.add_argument('--valid_size', default=0.1, help='')
parser.add_argument('--epochs', default=150, help='')
parser.add_argument('--controller_step', default=30, help='')
parser.add_argument('--controller_aggregate', default=10, help='')
parser.add_argument('--dropout', default=0.9, help='dropout rate')
parser.add_argument('--use_auxiliary', default=False, action='store_true', help='auxiliary loss for child.')

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
	                      num_workers=int(args.num_worker), sampler=train_sampler, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, 
	                      num_workers=int(args.num_worker), sampler=valid_sampler, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=100, 
	                     shuffle=False, num_workers=int(args.num_worker), pin_memory=True)

# there are 10 classes so the dataset name is cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
		   'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Making model..')
controller = Controller().to(device)
criterion = nn.CrossEntropyLoss()
controller_optimizer = optim.Adam(controller.parameters(), lr=0.0035)


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    grad_norm = total_norm ** (1. / 2)
    return grad_norm

    
def train_child(epoch, controller, child, child_optimizer):
    child.train()
    controller.eval()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        controller.init_hidden(batch_size=1)
        normal_arc, reduction_arc, _, _ = controller.sample_child()  # sample architecture 
        outputs, aux_outs = child(inputs, normal_arc, reduction_arc)  # forward with sampled arch

        loss = criterion(outputs, targets)
        if args.use_auxiliary:
            loss += 0.4 * criterion(aux_outs, targets)

        child_optimizer.zero_grad()
        loss.backward()
        grad_norm = get_grad_norm(child)
        torch.nn.utils.clip_grad_norm_(child.parameters(), 5.0)
        child_optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('train child epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f} | grad norm: {:.3f}'.format(epoch, batch_idx, 
                len(train_loader), train_loss/(batch_idx+1), 100.*correct/total, grad_norm))

    return child


def train_controller(controller, child, running_reward):
    child.eval()
    controller.train()

    # todo: in paper controller has to be updated for 2000 times.
    # but there is no information about mini-batch size. Need to be checked
    valid_iterator = iter(valid_loader)
    for batch_idx in range(int(args.controller_step)):
        try:
            inputs, targets = next(valid_iterator)
        except StopIteration:
            valid_iterator = iter(valid_loader)
            inputs, targets = next(valid_iterator)

        controller_optimizer.zero_grad()

        # accumulate gradients
        reward_list = list()
        for _ in range(args.controller_aggregate):
            controller.init_hidden(batch_size=1)
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
            reward_list.append(reward)

            # 2. using the reward, train controller with REINFORCE
            baseline = 0.99 * running_reward + 0.01 * reward
            log_prob = torch.cat(log_prob_seq, dim=0).sum()
            entropy = torch.cat(entropy_seq, dim=0).sum()

            loss = - log_prob * (reward - baseline)
            loss = loss - 0.0001 * entropy.detach()
            loss = loss / args.controller_aggregate

            loss.backward(retain_graph=True)
        
        grad_norm = get_grad_norm(controller)
        controller_optimizer.step()
        running_reward = 0.99 * running_reward + 0.01 * np.mean(reward_list)

        if batch_idx % 10 == 0:
            print('train controller : [{}/{}]| loss: {:.3f} | running reward: {:.3f} | entropy: {:.3f} |  grad norm: {:.3f}'.format(batch_idx, 
                args.controller_step, loss.item(), running_reward, entropy.item(), grad_norm))

    return controller, running_reward


def test_child(model):
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


def main(controller, cosine_annealing_scheduler):
	running_reward = 0
	child = Child(dropout_rate=float(args.dropout), use_auxiliary=args.use_auxiliary).to(device)
	
	child_optimizer = optim.SGD(child.parameters(), lr=0.05, 
		                      momentum=0.9, weight_decay=1e-4, nesterov=True)
	cosine_lr_scheduler = cosine_annealing_scheduler(child_optimizer, lr_max=0.05, lr_min=0.0005)
			
	for epoch in range(150):
		cosine_lr_scheduler.step()
		child = train_child(epoch, controller, child, child_optimizer)
		controller, running_reward = train_controller(controller, child, running_reward)
		
		torch.save(child.state_dict(), './save_model/child.pth')
		torch.save(controller.state_dict(), './save_model/controller.pth')

	child_acc = test_child(child)
	print('training is done. Child accuarcy is ', child_acc)


if __name__=="__main__":
	main(controller, cosine_annealing_scheduler)
