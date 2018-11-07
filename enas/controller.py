import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Controller(nn.Module):
	def __init__(self):
		super(Controller, self).__init__()
		self.num_nodes = 7
		self.lstm_size = 64
		self.embed = nn.Embedding(num_embeddings=2+5, embedding_dim=self.lstm_size)
		self.lstm = nn.LSTMCell(input_size=self.lstm_size, hidden_size=self.lstm_size)
		## todo: add initializer for lstm
		self.lstm.bias_ih.data.fill_(0)
		self.lstm.bias_hh.data.fill_(0)
		self.hx, self.cx = self.init_hidden(batch_size=1)
		# 2 indices and 5 operations
		self.fc_index = nn.Linear(in_features=self.lstm_size, out_features=2)
		self.fc_ops = nn.Linear(in_features=self.lstm_size, out_features=5)

		self.temperature = 5.0
		self.tanh_constant = 2.5

	def reset_parameters(self):
		pass

	def init_hidden(self, batch_size):
		hx = torch.zeros(batch_size, self.lstm_size).to(device)
		cx = torch.zeros(batch_size, self.lstm_size).to(device)
		return (hx, cx)

    # anchor is a placeholder for saving previous cell's lstm output
    # anchor_w is little different from anchor. The linear transformation of lstm output is saved 
	def forward(self, inputs, arc_seq, entropy_list, log_prob_list):
		# sample 2 indices for skip connection
		# this needs previous layer indices
		
		for i in range(2):
			if self.is_first == True and i == 0:
				emb_x = inputs
				self.is_first = False
			else:
				emb_x = self.embed(inputs)

			hidden = (self.hx, self.cx)
			self.hx, self.cx = self.lstm(emb_x, hidden)
			logits = self.fc_index(self.hx)
			logits /= self.temperature
			logits = self.tanh_constant*F.tanh(logits)
			probs = F.softmax(logits, dim=-1)
			log_prob = F.log_softmax(logits, dim=-1)
			action = torch.multinomial(probs, 1)[0]
			arc_seq.append(action)
			
			selected_log_prob = log_prob[:, action.long()]
			entropy = -(log_prob * probs).sum(1, keepdim=False)
			entropy_list.append(entropy)
			log_prob_list.append(log_prob)

			inputs = action

		# sample 2 operations for computation
		for i in range(2):
			emb_x = self.embed(inputs)
			hidden = (self.hx, self.cx)
			self.hx, self.cx = self.lstm(emb_x, hidden)
			logits = self.fc_ops(self.hx)
			logits /= self.temperature
			logits = self.tanh_constant*F.tanh(logits)

			probs = F.softmax(logits, dim=-1)
			log_prob = F.log_softmax(logits, dim=-1)
			action = torch.multinomial(probs, 1)[0]
			arc_seq.append(action)
			
			selected_log_prob = log_prob[:, action.long()]
			entropy = -(log_prob * probs).sum(1, keepdim=False)
			entropy_list.append(entropy)
			log_prob_list.append(log_prob)

			inputs = action + 2

		return inputs, arc_seq, entropy_list, log_prob_list


	# sample child model specifications
	# this is micro controller so sample architecture for 2 cells(normal, reduction)
	def sample_child(self):
		# for each node, there is 4 indices for constructing architecture of the node. 
		# 2 previous node indices and 2 operation indices
		normal_arc, reduction_arc = [], []
		# entropy and log prob is for the training of controller
		entropy_list, log_prob_list = [], []
		
		# sample normal architecture
		inputs = torch.zeros(1, self.lstm_size)
		self.is_first = True
		for _ in range(self.num_nodes-2):
			outputs = self.forward(inputs, normal_arc, entropy_list, log_prob_list)
			inputs, normal_arc, entropy_list, log_prob_list = outputs

		# sample reduction architecture
		for _ in range(self.num_nodes-2):
			outputs = self.forward(inputs, reduction_arc, entropy_list, log_prob_list)
			inputs, reduction_arc, entropy_list, log_prob_list = outputs

		return normal_arc, reduction_arc, entropy_list, log_prob_list
