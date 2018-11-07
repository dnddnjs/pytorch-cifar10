import torch
import torch.nn as nn
import torch.nn.functional as F

class Child(nn.Module):
	def __init__(self):
		super(Child).__init__()
		self.embed = nn.Embedding()