import torch.nn as nn


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, downsample=None):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
			                   stride=stride, padding=1, bias=False) 
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
			                   stride=1, padding=1, bias=False) 
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, layers, num_classes=10):
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.relu = nn.ReLU(inplace=True)

		# feature map size = 32x32x16
		layers_list = nn.ModuleList(
			[ResidualBlock(16, 16, stride=1)] * layers[0]
			)
		self.layers_2n = nn.Sequential(*layers_list)

		# feature map size = 16x16x32
		downsample = nn.Sequential(
				     	nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
				     	nn.BatchNorm2d(32))
		layers_list = nn.ModuleList(
			[ResidualBlock(16, 32, stride=2, downsample=downsample)] + \
			[ResidualBlock(32, 32, stride=1)] * (layers[1] - 1)
			)
		self.layers_4n = nn.Sequential(*layers_list)

		# feature map size = 8x8x64
		downsample = nn.Sequential(
				     	nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
				     	nn.BatchNorm2d(64))
		layers_list = nn.ModuleList(
			[ResidualBlock(32, 64, stride=2, downsample=downsample)] + \
			[ResidualBlock(64, 64, stride=1)] * (layers[1] - 1)
			)
		self.layers_6n = nn.Sequential(*layers_list)

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
	model = ResNet([9, 9, 9]) 
	return model
