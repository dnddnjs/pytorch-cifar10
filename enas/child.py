import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SkippingBranch(nn.Module):
	def __init__(self, planes, stride=2):
		super(SkippingBranch, self).__init__()
		self.conv1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, 
								padding=0, bias=False)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, 
								padding=0, bias=False)
		self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=stride, padding=0)    

	def forward(self, x):
		out1 = self.conv1(self.avg_pool(x))
		shift_x = x[:, :, 1:, 1:]
		shift_x= F.pad(shift_x, (0, 1, 0, 1))
		out2 = self.conv2(self.avg_pool(shift_x))
		out = torch.cat([out1, out2], dim=1)
		return out


def conv3x3(planes, stride=1):
	separable_conv = nn.Sequential(
						nn.Conv2d(planes, planes, kernel_size=3, stride=stride, 
						padding=1, groups=planes, bias=False),
						nn.Conv2d(planes, planes, kernel_size=1, stride=stride, 
						padding=0, bias=False))
	return separable_conv


def conv5x5(planes, stride=1):
	separable_conv = nn.Sequential(
						nn.Conv2d(planes, planes, kernel_size=5, stride=stride, 
						padding=2, groups=planes, bias=False),
						nn.Conv2d(planes, planes, kernel_size=1, stride=stride, 
						padding=0, bias=False))
	return separable_conv


def enas_conv(planes, kernel, stride=1):
	if kernel == 3:
		conv = conv3x3
	else:
		conv = conv5x5

	stack_conv = nn.Sequential(
			nn.ReLU(inplace=False),
			nn.BatchNorm2d(planes),
			conv(planes, stride),

			nn.ReLU(inplace=False),
			nn.BatchNorm2d(planes),
			conv(planes, stride)
		)
	return stack_conv


class Node(nn.Module):
	def __init__(self, planes, stride=1):
		super(Node, self).__init__()
		self.conv3x3 = enas_conv(planes, 3, stride)
		self.conv5x5 = enas_conv(planes, 5, stride)
		self.avgpool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

	def forward(self, x, op_id):
		print(x.size())
		out = [self.conv3x3(x), self.conv5x5(x), self.avgpool(x), self.maxpool(x), x]
		out = out[op_id]
		return out


class Cell(nn.Module):
	def __init__(self, arc, planes, stride=2):
		super(Cell, self).__init__()
		self.planes = planes
		self.arc = arc
		self.node_list = nn.ModuleList([Node(self.planes)]*10)
		self.conv_list = nn.ModuleList([nn.Conv2d(self.planes, self.planes, kernel_size=1, stride=1, 
						padding=0, bias=False)]*7)
		self.bn = nn.BatchNorm2d(self.planes)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, prev_outputs):
		# make 5 node = 1 cell
		layers = prev_outputs
		node_used = []
		for i in range(5):
			prev_outputs = torch.stack(layers)
			node_arc = self.arc[i*4:(i+1)*4]
			x_id = node_arc[0]
			x_op = node_arc[2]
			x = prev_outputs[x_id, :, : ,:].squeeze(0)
			x = self.node_list[2*i](x, x_op)
			x_one_hot = torch.zeros(7)
			x_one_hot.scatter_(0, x_id, 1)

			y_id = node_arc[1]
			y_op = node_arc[3]
			y = prev_outputs[y_id, :, : ,:].squeeze(0)
			y = self.node_list[2*i+1](y, y_op)
			y_one_hot = torch.zeros(7)
			y_one_hot.scatter_(0, y_id, 1)

			out = x + y
			layers.append(out)
			node_used.extend([x_one_hot, y_one_hot])

		# find output node which is never used as input of the other nodes
		used_stack = torch.stack(node_used)
		node_used = torch.sum(used_stack, dim=0)
		out_index = (node_used == 0).nonzero()
		out = torch.stack(layers).squeeze(0)
		out = self.relu(out)

		num_out = out_index.size(0)
		conv_outputs = []
		for i in range(num_out):
			index = out_index[i]
			out_i = out[i].squeeze()
			out_conv = self.conv_list[index](out_i)
			conv_outputs.append(out_conv)

		out = torch.stack(conv_outputs).sum(0)
		out = self.bn(out)
		return out


class Child(nn.Module):
	def __init__(self, normal_arc, reduction_arc, num_classes=10):
		super(Child, self).__init__()
		self.num_filters = 20
		self.conv1 = nn.Conv2d(3, self.num_filters, kernel_size=3, stride=1, 
			                   padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(self.num_filters)
		self.relu = nn.ReLU(inplace=False)
		self.avg_pool = nn.AvgPool2d(8, stride=1)
		self.fc_out = nn.Linear(80, num_classes)
		
		self.normal_arc = normal_arc
		self.reduction_arc = reduction_arc
		# if architecture change..?
		self.cell_list = nn.ModuleList(
				[Cell(self.normal_arc, self.num_filters)]*2 + \
				[Cell(self.reduction_arc, 2*self.num_filters)] + \
				[Cell(self.normal_arc, 2*self.num_filters)]*2 + \
				[Cell(self.reduction_arc, 4*self.num_filters)] + \
				[Cell(self.normal_arc, 4*self.num_filters)]*2
			)

	def forward(self, x):
		self.num_filters = 20
		# make first two input
		x = self.relu(self.bn1(self.conv1(x)))
		cell_outputs = [x, x]

		# 1-3 cells -----------------------------------------------------
		cell_id = 0
		for i in range(2):
			x = self.cell_list[cell_id](cell_outputs)
			cell_outputs = [cell_outputs[-1], x]
			cell_id += 1
		
		reduce_input = SkippingBranch(planes=self.num_filters)
		self.num_filters *= 2
		# todo: after reducing input, 
		cell_outputs = [reduce_input(cell_outputs[0]), reduce_input(cell_outputs[1])]
		x = self.cell_list[cell_id](cell_outputs)
		cell_outputs = [cell_outputs[-1], x]
		cell_id += 1

		# 4-5 cells -----------------------------------------------------
		for i in range(2):
			x = self.cell_list[cell_id](cell_outputs)
			cell_outputs = [cell_outputs[-1], x]
			cell_id += 1
		
		reduce_input = SkippingBranch(planes=self.num_filters)
		self.num_filters *= 2
		# todo: after reducing input, 
		cell_outputs = [reduce_input(cell_outputs[0]), reduce_input(cell_outputs[1])]
		x = self.cell_list[cell_id](cell_outputs)
		cell_outputs = [cell_outputs[-1], x]
		cell_id += 1

		# 6-7 cells -----------------------------------------------------
		for i in range(2):
			x = self.cell_list[cell_id](cell_outputs)
			cell_outputs = [cell_outputs[-1], x]
			cell_id += 1
		
		# output cell
		x = self.avg_pool(x)
		x = x.view(x.size(0), -1)
		x = self.fc_out(x)
		return x
