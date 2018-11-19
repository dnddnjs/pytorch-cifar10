import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ReduceBranch(nn.Module):
	def __init__(self, planes, stride=2):
		super(ReduceBranch, self).__init__()
		self.conv1 = nn.Conv2d(planes, planes, kernel_size=1, 
							   stride=1, padding=0, bias=False)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, 
							   stride=1, padding=0, bias=False)
		self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=stride, padding=0)    

	def forward(self, x):
		out1 = self.conv1(self.avg_pool(x))
		shift_x = x[:, :, 1:, 1:]
		shift_x= F.pad(shift_x, (0, 1, 0, 1))
		out2 = self.conv2(self.avg_pool(shift_x))
		out = torch.cat([out1, out2], dim=1)
		return out


def enas_conv(planes, kernel, stride=1):
	seperable_conv3x3 = nn.Sequential(
		nn.Conv2d(planes, planes, kernel_size=3, stride=stride, 
				  padding=1, groups=planes, bias=False),
		nn.Conv2d(planes, planes, kernel_size=1, stride=stride, 
				  padding=0, bias=False))
	
	seperable_conv5x5 = nn.Sequential(
		nn.Conv2d(planes, planes, kernel_size=5, stride=stride, 
				  padding=2, groups=planes, bias=False),
		nn.Conv2d(planes, planes, kernel_size=1, stride=stride, 
				  padding=0, bias=False))

	conv = seperable_conv3x3 if kernel == 3 else seperable_conv5x5

	stack_conv = nn.Sequential(
		nn.ReLU(inplace=False), 
		nn.BatchNorm2d(planes), 
		conv,
		
		nn.ReLU(inplace=False), 
		nn.BatchNorm2d(planes), 
		conv
	)
	return stack_conv


class Node(nn.Module):
	def __init__(self, planes, stride=1):
		super(Node, self).__init__()
		conv3x3 = enas_conv(planes, 3, stride)
		conv5x5 = enas_conv(planes, 5, stride)
		avgpool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
		maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
		self.ops = nn.ModuleList([conv3x3, conv5x5, avgpool, maxpool])

	def forward(self, x, op_id):
		# op_id 4 is for the identity mapping. Identity mapping can't be included by nn.modulelist
		if op_id == 4:
			out = x
		else:
			out = self.ops[op_id](x)
		return out
	

class Cell(nn.Module):
	def __init__(self, planes, stride=2):
		super(Cell, self).__init__()
		self.planes = planes
		self.node_list = nn.ModuleList([Node(self.planes) for _ in range(10)])
		self.conv_list = nn.ModuleList([nn.Conv2d(self.planes, self.planes, 
							kernel_size=1, stride=1, padding=0, bias=False) for _ in range(7)])
		self.bn = nn.BatchNorm2d(self.planes)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, prev_outputs, arc):
		# make 5 node = 1 cell
		layers = prev_outputs
		node_used = []
		for i in range(5):
			prev_outputs = torch.stack(layers)
			node_arc = arc[i*4:(i+1)*4]
			x_id = node_arc[0]
			x_op = node_arc[2]
			x = prev_outputs[x_id, :, : ,:].squeeze(0)
			x = self.node_list[2*i](x, x_op)
			x_one_hot = torch.zeros(7).to(device)
			x_one_hot.scatter_(0, x_id, 1)

			y_id = node_arc[1]
			y_op = node_arc[3]
			y = prev_outputs[y_id, :, : ,:].squeeze(0)
			y = self.node_list[2*i+1](y, y_op)
			y_one_hot = torch.zeros(7).to(device)
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
	def __init__(self, dropout_rate, num_classes=10, use_auxiliary=False):
		super(Child, self).__init__()
		self.num_filters = 20
		self.conv1 = nn.Conv2d(3, self.num_filters, kernel_size=3, stride=1, 
			                   padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(self.num_filters)
		self.relu = nn.ReLU(inplace=True)
		self.avg_pool = nn.AvgPool2d(8, stride=1)
		self.fc_out = nn.Linear(80, num_classes)
		self.dropout_rate = dropout_rate
		self.use_auxiliary = use_auxiliary
		
		self.cell_list = nn.ModuleList(
			[Cell(self.num_filters) for _ in range(2)] + \
			[Cell(2*self.num_filters)] + \
			[Cell(2*self.num_filters) for _ in range(2)] + \
			[Cell(4*self.num_filters)] + \
			[Cell(4*self.num_filters) for _ in range(2)]
		)

		self.reduce_module_list = nn.ModuleList(
			[ReduceBranch(planes=self.num_filters),
			ReduceBranch(planes=2*self.num_filters)]
		)
		
		if use_auxiliary:
			self.aux_conv = nn.Sequential(
				nn.Conv2d(80, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
				nn.Conv2d(128, 768, kernel_size=8), nn.BatchNorm2d(768), nn.ReLU(inplace=True)
			)
			self.aux_fc = nn.Linear(768, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', 
					                    nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
				
	def forward(self, x, normal_arc, reduction_arc):
		self.num_filters = 20
		# make first two input
		x = self.relu(self.bn1(self.conv1(x)))
		cell_outputs = [x, x]

		# 1-3 cells -----------------------------------------------------
		cell_id = 0
		for i in range(2):
			x = self.cell_list[cell_id](cell_outputs, normal_arc)
			cell_outputs = [cell_outputs[-1], x]
			cell_id += 1
		
		reduce_input = self.reduce_module_list[0]
		self.num_filters *= 2
		# todo: after reducing input, 
		cell_outputs = [reduce_input(cell_outputs[0]), reduce_input(cell_outputs[1])]
		x = self.cell_list[cell_id](cell_outputs, reduction_arc)
		cell_outputs = [cell_outputs[-1], x]
		cell_id += 1

		# 4-5 cells -----------------------------------------------------
		for i in range(2):
			x = self.cell_list[cell_id](cell_outputs, normal_arc)
			cell_outputs = [cell_outputs[-1], x]
			cell_id += 1
		
		reduce_input = self.reduce_module_list[1]
		self.num_filters *= 2
		# todo: after reducing input, 
		cell_outputs = [reduce_input(cell_outputs[0]), reduce_input(cell_outputs[1])]
		x = self.cell_list[cell_id](cell_outputs, reduction_arc)
		cell_outputs = [cell_outputs[-1], x]
		cell_id += 1

		# 6-7 cells -----------------------------------------------------
		for i in range(2):
			x = self.cell_list[cell_id](cell_outputs, normal_arc)
			cell_outputs = [cell_outputs[-1], x]
			cell_id += 1
		
		x = self.relu(x)
		aux = None
		if self.training and self.use_auxiliary:
			aux = self.aux_conv(x)
			aux = aux.view(aux.size(0), -1)
			aux = self.aux_fc(aux)
		
		# output cell
		x = self.avg_pool(x)
		x = F.dropout(x, self.dropout_rate, self.training, True)
		x = x.view(x.size(0), -1)
		x = self.fc_out(x)
		return x, aux
