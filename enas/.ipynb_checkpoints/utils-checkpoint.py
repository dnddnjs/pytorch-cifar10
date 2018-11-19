def _apply_drop_path(self, x, layer_id, keep_prob=0.6):

	layer_ratio = float(layer_id + 1) / (self.num_layers + 2)
	drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)

	step_ratio = tf.to_float(self.global_step + 1) / tf.to_float(self.num_train_steps)
	step_ratio = tf.minimum(1.0, step_ratio)
	drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)

	x = drop_path(x, drop_path_keep_prob)
	mask = torch.Tensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).to(DEVICE)
	x.div_(keep_prob)
	x.mul_(mask)
	return x

