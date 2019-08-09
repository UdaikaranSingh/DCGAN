import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import torch.optim as optim
from torchvision.datasets import ImageFolder
import random
import torchvision.transforms as transforms
import re

from model import Classifier, Decoder, Discriminator, Classifier

import numpy as np

#######################################
# Importing and setting up networks 
#######################################

from model import Encoder, Decoder, Classifier, Discriminator

cuda_available = torch.cuda.is_available()
encoded_size = 50
batch_size = 1

if cuda_available:
    print("Using GPU")
else:
    print("Using CPU")


if cuda_available:
	encoder = Encoder(batch_size, encoded_size).cuda()
	decoder = Decoder(batch_size, encoded_size).cuda()
	classifier = Classifier(batch_size, encoded_size).cuda()
	discriminator = Discriminator(batch_size, encoded_size).cuda()

	if os.path.exists("enocder_model_celeb.pth"):
		print("Loading in Model")
		encoder.load_state_dict(torch.load("encoder_model_celeb.pth"))
		decoder.load_state_dict(torch.load("decoder_model_celeb.pth"))
		classifier.load_state_dict(torch.load("classifier_model_celeb.pth"))
		discriminator.load_state_dict(torch.load("discriminator_model_celeb.pth"))
else:
	encoder = Encoder(batch_size, encoded_size)
	decoder = Decoder(batch_size, encoded_size)
	classifier = Classifier(batch_size, encoded_size)
	discriminator = Discriminator(batch_size, encoded_size)

	if os.path.exists("encoder_model_celeb.pth"):
		print("Loading in Model")
		encoder.load_state_dict(torch.load("encoder_model_celeb.pth"), map_location='cpu')
		decoder.load_state_dict(torch.load("decoder_model_celeb.pth"), map_location=torch.device('cpu'))
		classifier.load_state_dict(torch.load("classifier_model_celeb.pth"), map_location=torch.device('cpu'))
		discriminator.load_state_dict(torch.load("discriminator_model_celeb.pth"), map_location=torch.device('cpu'))

##################################################
# Definining Hyperparameters of Training Procedure
##################################################


transform = transforms.Compose([transforms.Resize((128, 128)),
    transforms.ToTensor()
])
dataset_path = os.path.dirname(os.getcwd()) + "/datasets/celeba"
training_data = ImageFolder(dataset_path, transform = transform)
data_loader = torch.utils.data.DataLoader(training_data,
                                          batch_size= batch_size,
                                          shuffle=True,
                                          num_workers= 1)


n = len(data_loader.dataset.imgs)

output = []
patt = "celeba/.*"
for step, (images, _) in enumerate(data_loader, 0):
    if (step % 1000 == 0):
        print(step / n)
    if cuda_available:
        imgs = Variable(images.type(torch.cuda.FloatTensor)) #real images
    else:
        imgs = Variable(images.type(torch.FloatTensor))
        
    z_pred = encoder(images).cpu().detach().numpy()
    address = re.findall(patt,data_loader.dataset.imgs[step][0])
    
    output.append((z_pred, address))


for i in range(len(output)):
	output[i] = list(output[i])

for i in range(len(output)):
    output[i][0] = output[i][0].reshape(50,)

np.save("output_celeba.npy", np.array(output))