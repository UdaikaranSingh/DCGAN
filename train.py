import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import torch.optim as optim
from torchvision.datasets import ImageFolder
import random
import torchvision.transforms as transforms

#######################################
# Importing and setting up networks 
#######################################

from model import Encoder, Decoder, Classifier, Discriminator

cuda_available = torch.cuda.is_available()
encoded_size = 50
batch_size = 16

if cuda_available:
    print("Using GPU")
else:
    print("Using CPU")


if cuda_available:
	encoder = Encoder(batch_size, encoded_size).cuda()
	decoder = Decoder(batch_size, encoded_size).cuda()
	classifier = Classifier(batch_size, encoded_size).cuda()
	discriminator = Discriminator(batch_size, encoded_size).cuda()

	if os.path.exists("enocder_model.pth"):
		print("Loading in Model")
		encoder.load_state_dict(torch.load("encoder_model.pth"))
		decoder.load_state_dict(torch.load("decoder_model.pth"))
		classifier.load_state_dict(torch.load("classifier_model.pth"))
		discriminator.load_state_dict(torch.load("discriminator_model.pth"))
else:
	encoder = Encoder(batch_size, encoded_size)
	decoder = Decoder(batch_size, encoded_size)
	classifier = Classifier(batch_size, encoded_size)
	discriminator = Discriminator(batch_size, encoded_size)

	if os.path.exists("enocder_model.pth"):
		print("Loading in Model")
		encoder.load_state_dict(torch.load("encoder_model.pth"))
		decoder.load_state_dict(torch.load("decoder_model.pth"))
		classifier.load_state_dict(torch.load("classifier_model.pth"))
		discriminator.load_state_dict(torch.load("discriminator_model.pth"))

##################################################
# Definining Hyperparameters of Training Procedure
##################################################

random.seed(1)
learning_rate = 0.2e-4
beta1 = 0.5
beta2 = 0.999
num_epochs = 10
epsilon = 1e-8

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) 
	+ list(classifier.parameters()), 
	lr=learning_rate, 
	betas=(beta1, beta2))

transform = transforms.Compose([transforms.Resize((128, 128)),
    transforms.ToTensor()
])
dataset_path = os.path.dirname(os.getcwd()) + "/datasets/celeba"
training_data = ImageFolder(dataset_path, transform = transform)
data_loader = torch.utils.data.DataLoader(training_data,
                                          batch_size= batch_size,
                                          shuffle=True,
                                          num_workers= 1)

l2_loss = nn.MSELoss()
BCE_loss = nn.BCELoss()

##################################################
# Training Procedure
##################################################

loss_tracker = []

for epoch in range(num_epochs):

	count = 0

	print("epoch: " + str(epoch + 1))

	for images, steps in data_loader:

		if (images.shape[0] != batch_size):
			break

		elif (count% 2 == 0):
			x1 = images

		else: 
			#case of steps % 2 == 1
			encoder.zero_grad()
			decoder.zero_grad()
			classifier.zero_grad()

			x2 = images

			f1 = encoder(x1)
			f2 = encoder(x2)

			m = torch.FloatTensor(batch_size, encoded_size).uniform_(0, 1)

			f12 = m * f1 + (torch.ones(batch_size, encoded_size) - m) * f2

			x3 = decoder(f12)
			f3 = encoder(x3)

			f31 = m * f3 + (torch.ones(batch_size, encoded_size) - m) * f1

			x4 = decoder(f31)

			loss_l2 = l2_loss(x1, x4)
			dicsrimination_loss = torch.mean(torch.log(discriminator(x1) + epsilon) + torch.log(1 - discriminator(x3) + epsilon))
			x_cat = torch.cat((x1, x2, x3), dim = 1)
			classifer_loss = BCE_loss(torch.sigmoid(classifier(x_cat)), m)

			#backpropagation steps & optimization steps
			loss_l2.backward(retain_graph = True)
			dicsrimination_loss.backward(retain_graph = True)
			classifer_loss.backward()
			optimizer.step()
		count = count + 1

	#saving models:
	path = "decoder_mdoel.pth"
	torch.save(decoder.state_dict(), path)
	path = "encoder_model.pth"
	torch.save(encoder.state_dict(), path)
	path = "classifier_model.pth"
	torch.save(classifier.state_dict(), path)
	path = "discriminator_model.pth"
	torch.save(discriminator.state_dict(), path)







