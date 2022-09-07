import numpy as np
import torch
from torch.autograd import Variable


class LinearRegression(torch.nn.Module):
	def __init__(self, input_size, output_size, bias):
		super(LinearRegression, self).__init__()
		self.linear = torch.nn.Linear(input_size, output_size, bias=bias)

	def forward(self, x):
		out = self.linear(x)
		return out


def best_fit(x, y):
	input_size, output_size = 1, 1
	model = LinearRegression(input_size, output_size, bias=False)
	learning_rate = 0.000001
	epochs = 100

	if torch.cuda.is_available():
		model.cuda()

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

	for epoch in range(epochs):
		# Converting inputs and labels to Variable
		if torch.cuda.is_available():
			inputs = Variable(x.cuda())
			labels = Variable(y.cuda())
		else:
			inputs = Variable(x)
			labels = Variable(y)

		# Clear gradient buffers because we don't want any gradient from previous epoch to carry forward,
		# don't want to cumulate gradients
		optimizer.zero_grad()

		# get output from the model, given the inputs
		outputs = model(inputs)

		# get loss for the predicted output
		loss = criterion(outputs, labels)
		print(loss)
		# get gradients w.r.t to parameters
		loss.backward()

		# update parameters
		optimizer.step()

	# Return slope as tensor
	return model.linear.weight
