import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Controller(nn.Module):
	def __init__(self):
		super(Controller, self).__init__()
		self.num_nodes = 7
		self.lstm_size = 64
		self.embed_first = nn.Embedding(num_embeddings=1, embedding_dim=self.lstm_size)
		self.embed_ops = nn.Embedding(num_embeddings=5, embedding_dim=self.lstm_size)
		self.lstm = nn.LSTMCell(input_size=self.lstm_size, hidden_size=self.lstm_size)
		## todo: add initializer for lstm
		# self.lstm.bias_ih.data.fill_(0)
		# self.lstm.bias_hh.data.fill_(0)

		self.hx, self.cx = self.init_hidden(batch_size=1)
		# fully-connected layers for index of previous cell outputs
		self.fc_index_prev = nn.Linear(in_features=self.lstm_size, out_features=self.lstm_size)
		self.fc_index_curr = nn.Linear(in_features=self.lstm_size, out_features=self.lstm_size)
		self.fc_index_out = nn.Linear(in_features=self.lstm_size, out_features=1)
		# fully-connected layer for 5 operations
		self.fc_ops = nn.Linear(in_features=self.lstm_size, out_features=5)
		self.init_parameters()

		self.temperature = 1.10
		self.tanh_constant = 2.5

	def init_parameters(self):
		init_range = 0.1
		for param in self.lstm.parameters():
			param.data.uniform_(-init_range, init_range)

		self.fc_index_prev.bias.data.fill_(0)
		self.fc_index_curr.bias.data.fill_(0)
		# print(self.fc_ops.weight.size())
		# print(self.fc_ops.bias.size())
		init_bias = [10, 10, 0, 0, 0]
		self.fc_ops.bias.data = torch.Tensor(np.array(init_bias))
		

	def init_hidden(self, batch_size):
		hx = torch.zeros(batch_size, self.lstm_size).to(device)
		cx = torch.zeros(batch_size, self.lstm_size).to(device)
		return (hx, cx)

	# anchor is a placeholder for saving previous cell's lstm output
	# anchor_w is little different from anchor. The linear transformation of lstm output is saved 
	def sample_cell(self, arc_seq, entropy_list, log_prob_list):
		# lstm should have a dynamic size of output for indices of previous layer.
		# so save previous lstm outputs and fc outputs as a list
		prev_lstm_outputs, prev_fc_outputs = [], []
		inputs = torch.zeros(1).long().to(device)
		inputs = self.embed_first(inputs)
		for node_id in range(2):
			hidden = (self.hx, self.cx)
			self.hx, self.cx = self.lstm(inputs, hidden)
			prev_lstm_outputs.append(torch.zeros_like(self.hx))
			prev_fc_outputs.append(self.fc_index_prev(self.hx.clone()))			

		for node_id in range(2, self.num_nodes):
			# indices for previous nodes
			indices = torch.range(0, node_id).long()

			# sample 2 indices to select input of the node
			for i in range(2):
				hidden = (self.hx, self.cx)
				self.hx, self.cx = self.lstm(inputs, hidden)
				# todo: need to be fixed
				logits = self.fc_index_curr(self.hx)
				query = torch.cat(prev_fc_outputs)
				query = F.tanh(query + logits)
				query = self.fc_index_out(query)
				logits = query.view(query.size(-1), -1)

				logits = logits / self.temperature
				logits = self.tanh_constant*F.tanh(logits)
				probs = F.softmax(logits, dim=-1)
				log_prob = F.log_softmax(logits, dim=-1)
				action = torch.multinomial(probs, 1)[0]
				arc_seq.append(action)
				
				selected_log_prob = log_prob[:, action.long()]
				entropy = -(log_prob * probs).sum(1, keepdim=False)
				entropy_list.append(entropy)
				log_prob_list.append(log_prob)
				# next input for lstm is the output of selected previous node index
				inputs = prev_lstm_outputs[action]

			# sample 2 operations for computation
			for i in range(2):
				hidden = (self.hx, self.cx)
				self.hx, self.cx = self.lstm(inputs, hidden)
				logits = self.fc_ops(self.hx)
				logits /= self.temperature
				logits = self.tanh_constant*F.tanh(logits)
				init_bias = [0.25, 0.25, -0.25, -0.25, -0.25]
				logits += torch.Tensor(np.array(init_bias))

				probs = F.softmax(logits, dim=-1)
				log_prob = F.log_softmax(logits, dim=-1)
				action = torch.multinomial(probs, 1)[0]
				arc_seq.append(action)
				
				selected_log_prob = log_prob[:, action.long()]
				entropy = -(log_prob * probs).sum(1, keepdim=False)
				entropy_list.append(entropy)
				log_prob_list.append(log_prob)

				inputs = self.embed_ops(action)

			hidden = (self.hx, self.cx)
			self.hx, self.cx = self.lstm(inputs, hidden)
			prev_lstm_outputs.append(self.hx.clone())
			prev_fc_outputs.append(self.fc_index_prev(self.hx.clone()))
			inputs = torch.zeros(1).long()
			inputs = self.embed_first(inputs)
		return arc_seq, entropy_list, log_prob_list


	# sample child model specifications
	# this is micro controller so sample architecture for 2 cells(normal, reduction)
	def sample_child(self):
		# for each node, there is 4 indices for constructing architecture of the node. 
		# 2 previous node indices and 2 operation indices
		normal_arc, reduction_arc = [], []
		# entropy and log prob is for the training of controller
		entropy_list, log_prob_list = [], []
		
		# sample normal architecture
		outputs = self.sample_cell(normal_arc, entropy_list, log_prob_list)
		normal_arc, entropy_list, log_prob_list = outputs

		# sample reduction architecture
		outputs = self.sample_cell(reduction_arc, entropy_list, log_prob_list)
		reduction_arc, entropy_list, log_prob_list = outputs

		return normal_arc, reduction_arc, entropy_list, log_prob_list
