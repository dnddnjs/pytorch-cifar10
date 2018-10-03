from resnet import model
from cifar_data import CIFAR10

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=False, help='')
parser.add_argument('--batch_size', default=128, help='')
parser.add_argument('--num_worker', default=2, help='')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')
transforms_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transforms_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset_train = CIFAR10(root='./data', train=True, download=True, transform=transforms_train)
dataset_test = CIFAR10(root='./data', train=False, download=True, transform=transforms_test)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, 
	                      shuffle=True, num_workers=args.num_workers)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=100, 
	                     shuffle=True, num_workers=args.num_workers)

# there are 10 classes so the dataset name is cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
	       'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Making model..')
net = model.resnet18()
net = net.to(device)
if device == 'cuda':
	net = cuda.nn.DataParallel(net)
	cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

def train(epoch):
	net.train()

	train_loss = 0
	correct = 0
	total = 0

	for batch_idx, (inputs, targets) in enumerate(train_loader):
		inputs = inputs.to(device)
		targets = targets.to(device)
		outputs = net(inputs)
		loss = criterion(outputs, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = output.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

		progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
		    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def test(epoch):
	net.eval()

	test_loss = 0
	correct = 0
	total = 0

	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(train_loader):
			inputs = inputs.to(device)
			targets = targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = output.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

	acc = 100 * correct / total
	if acc > best_acc:
		print('==> Saving model..')
		state = {
		    'net': net.state_dict(),
		    'acc': acc,
		    'epoch': epoch,
		}
		if not os.path.isdir('save_model'):
		    os.mkdir('save_model')
		torch.save(state, './save_model/ckpt.pth')
		best_acc = acc


if __name__=='__main__':
	best_acc = 0
	for epoch in range(200):
		train(epoch)
		test(epoch)