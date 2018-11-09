import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)

def conv(kernel, planes):
    if kernel == 3:
        _conv = conv3x3
    elif kernel == 5:
        _conv = conv5x5
    else:
        raise NotImplemented(f"Unkown kernel size: {kernel}")

    return nn.Sequential(
            nn.ReLU(inplace=True),
            _conv(planes, planes),
            nn.BatchNorm2d(planes),
    )

class Child(nn.Module):
	def __init__(self, num_classes=10):
		super(Child, self).__init__()
		self.avg_pool = nn.AvgPool2d(8, stride=1)
		self.fc_out = nn.Linear(64, num_classes)

	def enas_cell(self, normal_arc, reduction_arc):
		return normal_cell, reduction_cell

	def build_model(self, normal_arc, reduction_arc):
		outputs = self.enas_cell(normal_arc, reduction_arc)
		normal_cell, reduction_cell = outputs


		return model

	def forward(self, x):
		cell_outputs = []
		
		for layer in range(num_layers):
			for i in range(num_normal_cells):
				x = normal_cell(x, cell_outputs)
				cell_outputs.append(x)
			x = reduction_cell(x, cell_outputs)
			cell_outputs.append(x)
		
		x = self.avg_pool(x)
		x = x.view(x.size(0), -1)
		x = self.fc_out(x)
		return x
