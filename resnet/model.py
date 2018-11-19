import torch.nn as nn
import torch.nn.functional as F

# code from https://github.com/KellerJordan/ResNet-PyTorch-CIFAR10/blob/master/model.py
class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityPadding, self).__init__()

        self.identity = nn.MaxPool2d(1, stride=stride)
        self.num_zeros = out_channels - in_channels
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out
	
	
class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
			                   stride=stride, padding=1, bias=False) 
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
			                   stride=1, padding=1, bias=False) 
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.stride = stride
		
		if down_sample:
			self.down_sample = IdentityPadding(in_channels, out_channels, stride)
		else:
			self.down_sample = None


	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.down_sample is not None:
			residual = self.down_sample(x)

		out += residual
		out = self.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, num_layers, block, num_classes=10):
		super(ResNet, self).__init__()
		self.num_layers = num_layers
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, 
							   stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.relu = nn.ReLU(inplace=True)
		
		# feature map size = 32x32x16
		self.layers_2n = self.get_layers(block, 16, 16, stride=1)
		# feature map size = 16x16x32
		self.layers_4n = self.get_layers(block, 16, 32, stride=2)
		# feature map size = 8x8x64
		self.layers_6n = self.get_layers(block, 32, 64, stride=2)

		# output layers
		self.avg_pool = nn.AvgPool2d(8, stride=1)
		self.fc_out = nn.Linear(64, num_classes)
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', 
					                    nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
	
	def get_layers(self, block, in_channels, out_channels, stride):
		if stride == 2:
			down_sample = True
		else:
			down_sample = False
		
		layers_list = nn.ModuleList(
			[block(in_channels, out_channels, stride, down_sample)])
			
		for _ in range(self.num_layers - 1):
			layers_list.append(block(out_channels, out_channels))

		return nn.Sequential(*layers_list)
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.layers_2n(x)
		x = self.layers_4n(x)
		x = self.layers_6n(x)

		x = self.avg_pool(x)
		x = x.view(x.size(0), -1)
		x = self.fc_out(x)
		return x


def resnet():
	block = ResidualBlock
	# total number of layers if 6n + 2. if n is 5 then the depth of network is 32.
	model = ResNet(5, block) 
	return model
