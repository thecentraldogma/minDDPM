# we feed in batches of data that have been corrupted with noise at various randomly chosen time-steps
# the 'data' here is a 1-dimensional sequence in the shape of a sinusoid with random initial phase
# the 'noise' here is a Gaussian corruption of the values of the input sequence. 
# 'samples' are therefore 1-dimensional sequences of a fixed length that should look like a sinuoid with random phase 

import numpy as np
import torch
import torch.nn as nn


def alphaT(t, beta): 
	alpha = 1-beta
	alphaT = alpha**(t)
	return alphaT

def forward_noise(x, t, beta):
	alpha_t = alphaT(beta, t)
	# Add Gaussian noise with mean 0 and covariance (1-alpha_t)*I to sqrt(alpha_t) * x
	noise = torch.randn(size = x.size()) * np.sqrt(1 - alpha_t)
	return (np.sqrt(alpha_t) * x) + noise

class DDPM(nn.Module): 

	def __init__(self, seq_len, beta):
		super().__init__()
		s = seq_len + 1
		self.seq_len = seq_len
		self.beta = beta
		self.linear1 = nn.Linear(s, int(s/1.5))
		self.linear2 = nn.Linear(int(s/1.5), int(s/2))
		self.linear3 = nn.Linear(int(s/2), int(s/3))
		self.linear4 = nn.Linear(int(s/3), int(s/2))
		self.linear5 = nn.Linear(int(s/2), seq_len)
		self.relu = nn.ReLU()


	def forward(self, x, t, target = None):
		# x and t are the inputs to the network. 
		# x is the noised data and t is the number of noise steps. 
		# the output is the predicted noise vector, the same length as x
		t1 = t.unsqueeze(dim = 1)
		x = torch.cat([x, t1], dim = 1)
		x = self.linear1(x)
		x = self.relu(x)
		x = self.linear2(x)
		x = self.relu(x)
		x = self.linear3(x)
		x = self.relu(x)
		x = self.linear4(x)
		x = self.relu(x)
		x = self.linear5(x)
		
		loss = None
		loss_func = torch.nn.MSELoss()
		if target is not None: 
			loss = loss_func(x, target) 

		return x, loss



	def sample(self): 
		# To sample, we start with a pure noise vector, x_t with t = T. 
		# and then we pass it through the forward process T times
		# the fwd process predicts the noise that's been added to the pure data sample x0 to get the corrupted sample
		# Subtracting the predicted noise gives an estimate of x0.
		# But we only subtract a part of this noise vector so as to take a step in the direction of the predictor x0. 
		# We then estimate the next prior sample x_{t-1} by scaling this point appropriately and adding an appropriate amount of noise
		x = torch.randn(self.seq_len)
		for t in range(T, 0, -1):
			predicted_noise = self.forward(x, t)
			alpha_t = 1 - self.beta
			alpha_t_bar = self.alphaT(t, self.beta)
			alpha_t_minus_one_bar = self.alphaT(t-1, self.beta)
			step_towards_predicted_x0 = (x - predicted_noise * (1-alpha_t) / np.sqrt(1 - alpha_t_bar))
			noise_variance = (1 - alpha_t_minus_one_bar)/(1 - alpha_t_bar) * self.beta
			if t > 1: 
				noise = torch.rand(self.seq_len)
			else: 
				noise = torch.zeros(self.seq_len)
			x = (1/np.sqrt(alpha_t)) * step_towards_predicted_x0 + (np.sqrt(noise_variance) * noise)
		return x


	def configure_optimizers(self, train_config):
		"""
		This long function is unfortunately doing something very simple and is being very defensive:
		We are separating out all parameters of the model into two buckets: those that will experience
		weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
		We are then returning the PyTorch optimizer object.
		"""

		# separate out all parameters to those that will and won't experience regularizing weight decay
		decay = set()
		no_decay = set()
		whitelist_weight_modules = (torch.nn.Linear, )
		blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
		for mn, m in self.named_modules():
			for pn, p in m.named_parameters():
				fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
				# random note: because named_modules and named_parameters are recursive
				# we will see the same tensors p many many times. but doing it this way
				# allows us to know which parent module any tensor p belongs to...
				if pn.endswith('bias'):
					# all biases will not be decayed	
					no_decay.add(fpn)
				elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
					# weights of whitelist modules will be weight decayed
					decay.add(fpn)
				elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
					# weights of blacklist modules will NOT be weight decayed
					no_decay.add(fpn)

		# validate that we considered every parameter
		param_dict = {pn: p for pn, p in self.named_parameters()}
		inter_params = decay & no_decay
		union_params = decay | no_decay
		assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
		assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
													% (str(param_dict.keys() - union_params), )

		# create the pytorch optimizer object
		optim_groups = [
		    {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
		    {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
		]
		optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
		return optimizer

