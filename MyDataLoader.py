import torch
from torch.utils import data

class Dataset(data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, features, labels, transform = None):
		'Initialization'
		self.labels = labels
		self.features = features
		self.transform = transform

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.features)

	def __getitem__(self, index):
		'Generates one sample of data'
		x = self.features[index]
		y = self.labels[index]
		if self.transform:
			x = self.transform(x)
		return x, y