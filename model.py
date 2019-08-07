import torch as torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    
    def __init__(self, batch_size, encoded_size):
        super(Encoder, self).__init__()
        
        #store variables
        self.batch_size = batch_size
        self.encoded_size = encoded_size
        self.neg_slope = 0.2
        
        #architecture
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size= 3)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= 3)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= 3)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.pooling = nn.MaxPool2d(kernel_size= 3, stride= 2)
        
        self.conv5 = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size= 3)
        self.bn5 = nn.BatchNorm2d(64)
        
        self.conv6 = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size= 3)
        self.bn6 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels= 96, kernel_size= 1)
        self.bn7 = nn.BatchNorm2d(96)
        
        self.conv8 = nn.Conv2d(in_channels=96, out_channels= 96, kernel_size= 3)
        self.bn8 = nn.BatchNorm2d(96)
        
        self.conv9 = nn.Conv2d(in_channels=96, out_channels= 96, kernel_size= 3)
        self.bn9 = nn.BatchNorm2d(96)
        
        self.conv10 = nn.Conv2d(in_channels=96, out_channels= 128, kernel_size= 1)
        self.bn10 = nn.BatchNorm2d(128)
        
        self.conv11 = nn.Conv2d(in_channels=128, out_channels= 128, kernel_size= 3)
        self.bn11 = nn.BatchNorm2d(128)
        
        self.conv12 = nn.Conv2d(in_channels=128, out_channels= 128, kernel_size= 3)
        self.bn12 = nn.BatchNorm2d(128)
        
        self.conv13 = nn.Conv2d(in_channels=128, out_channels= 160, kernel_size= 1)
        self.bn13 = nn.BatchNorm2d(160)
        
        self.conv14 = nn.Conv2d(in_channels=160, out_channels= 160, kernel_size= 1)
        self.bn14 = nn.BatchNorm2d(160)
        
        self.conv15 = nn.Conv2d(in_channels=160, out_channels= 160, kernel_size= 1)
        self.bn15 = nn.BatchNorm2d(160)
        
        self.conv16 = nn.Conv2d(in_channels=160, out_channels= 160, kernel_size= 1)
        self.bn16 = nn.BatchNorm2d(160)
        
        self.fully_connected = nn.Linear(in_features= 1440, out_features= self.encoded_size)
        
    def forward(self, x):
        
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn3(self.conv3(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn4(self.conv4(out)), negative_slope = self.neg_slope)
        
        out = self.pooling(out)

        out = F.leaky_relu(self.bn5(self.conv5(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn6(self.conv6(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn7(self.conv7(out)), negative_slope = self.neg_slope)
        
        out = self.pooling(out)

        out = F.leaky_relu(self.bn8(self.conv8(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn9(self.conv9(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn10(self.conv10(out)), negative_slope = self.neg_slope)
        
        out = self.pooling(out)

        out = F.leaky_relu(self.bn11(self.conv11(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn12(self.conv12(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn13(self.conv13(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn14(self.conv14(out)), negative_slope = self.neg_slope)
        
        out = self.pooling(out)

        out = F.leaky_relu(self.bn15(self.conv15(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn16(self.conv16(out)), negative_slope = self.neg_slope)
        
        out = out.view((self.batch_size, 1440))
        out = self.fully_connected(out)
        
        return out
        
class Decoder(nn.Module):
    
    def __init__(self, batch_size, encoded_size):
        super(Decoder, self).__init__()
        
        #storing variables
        self.batch_size = batch_size
        self.encoded_size = encoded_size
        self.neg_slope = 0.2
        
        #architecture
        self.fully_connected = nn.Linear(in_features= self.encoded_size, out_features= 4096)
        
        self.conv1 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.upsample = nn.Upsample(scale_factor= 2)
        
        self.conv3 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3)
        self.bn5 = nn.BatchNorm2d(64)
        
        self.conv6 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3)
        self.bn6 = nn.BatchNorm2d(64)
        
        self.conv7 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3)
        self.bn7 = nn.BatchNorm2d(64)
        
        self.conv8 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3)
        self.bn8 = nn.BatchNorm2d(64)
        
        self.conv9 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3)
        self.bn9 = nn.BatchNorm2d(64)
        
        self.conv10 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3)
        self.bn10 = nn.BatchNorm2d(64)
        
        self.conv11 = nn.Conv2d(in_channels= 64, out_channels= 3, kernel_size= 3)
        
        
        
        
    def forward(self, x):
        
        #takes in a batch of images (batch size x encoded size)
        
        out = self.fully_connected(x)
        out = out.view((self.batch_size, 64, 8, 8))
        out = F.leaky_relu(self.bn1(self.conv1(out)), negative_slope= self.neg_slope)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope= self.neg_slope)
        out = self.upsample(out)
        out = F.leaky_relu(self.bn3(self.conv3(out)), negative_slope= self.neg_slope)
        out = F.leaky_relu(self.bn4(self.conv4(out)), negative_slope= self.neg_slope)
        out = self.upsample(out)
        out = F.leaky_relu(self.bn5(self.conv5(out)), negative_slope= self.neg_slope)
        out = self.upsample(out)
        out = F.leaky_relu(self.bn6(self.conv6(out)), negative_slope= self.neg_slope)
        out = self.upsample(out)
        out = F.leaky_relu(self.bn5(self.conv7(out)), negative_slope= self.neg_slope)
        out = self.upsample(out)
        out = F.leaky_relu(self.bn6(self.conv8(out)), negative_slope= self.neg_slope)
        out = self.upsample(out)
        out = F.leaky_relu(self.bn5(self.conv9(out)), negative_slope= self.neg_slope)
        out = self.upsample(out)
        out = F.leaky_relu(self.bn6(self.conv10(out)), negative_slope= self.neg_slope)
        out = F.relu(self.conv11(out))
        
        return out
        
        
        
class Classifier(nn.Module):
    
    def __init__(self, batch_size, encoded_size):
        super(Classifier, self).__init__()
        
        #storing variables
        self.batch_size = batch_size
        self.encoded_size = encoded_size
        self.neg_slope = 0.2
        
        #architecture
        self.conv1 = nn.Conv2d(in_channels= 9, out_channels= 96, kernel_size= 8, stride= 2)
        self.bn1 = nn.BatchNorm2d(96)
        
        self.pooling = nn.MaxPool2d(kernel_size= 3, stride= 2)
        
        self.conv2 = nn.Conv2d(in_channels= 96, out_channels= 256, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(in_channels= 256, out_channels= 384, kernel_size= 3)
        self.bn3 = nn.BatchNorm2d(384)
        
        self.conv4 = nn.Conv2d(in_channels= 384, out_channels= 384, kernel_size= 3)
        self.bn4 = nn.BatchNorm2d(384)
        
        self.conv5 = nn.Conv2d(in_channels= 384, out_channels= 256, kernel_size= 3)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.fully_connected1 = nn.Linear(in_features= 1024, out_features= 4096)
        self.fully_connected2 = nn.Linear(in_features= 4096, out_features= 4096)
        self.fully_connected3 = nn.Linear(in_features= 4096, out_features= self.encoded_size)
        
    def forward(self, x):
        #x is size (batch size x 9 x 128 x 128)
        #note: stacking 3 rgb images
        
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope= self.neg_slope)
        
        out = self.pooling(out)
        
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope= self.neg_slope)
        
        out = self.pooling(out)
        
        out = F.leaky_relu(self.bn3(self.conv3(out)), negative_slope= self.neg_slope)
        
        out = F.leaky_relu(self.bn4(self.conv4(out)), negative_slope= self.neg_slope)
        
        out = F.leaky_relu(self.bn5(self.conv5(out)), negative_slope= self.neg_slope)
        out = self.pooling(out)
        
        out = out.view((self.batch_size, 1024))
        
        out = self.fully_connected1(out)
        out = self.fully_connected2(out)
        out = self.fully_connected3(out)
        
        return out

class Discriminator(nn.Module):
    
    def __init__(self, batch_size, encoded_size):
        super(Discriminator, self).__init__()
        
        #store variables
        self.batch_size = batch_size
        self.encoded_size = encoded_size
        self.neg_slope = 0.2
        
        #architecture
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size= 3)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= 3)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= 3)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.pooling = nn.MaxPool2d(kernel_size= 3, stride= 2)
        
        self.conv5 = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size= 3)
        self.bn5 = nn.BatchNorm2d(64)
        
        self.conv6 = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size= 3)
        self.bn6 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels= 96, kernel_size= 1)
        self.bn7 = nn.BatchNorm2d(96)
        
        self.conv8 = nn.Conv2d(in_channels=96, out_channels= 96, kernel_size= 3)
        self.bn8 = nn.BatchNorm2d(96)
        
        self.conv9 = nn.Conv2d(in_channels=96, out_channels= 96, kernel_size= 3)
        self.bn9 = nn.BatchNorm2d(96)
        
        self.conv10 = nn.Conv2d(in_channels=96, out_channels= 128, kernel_size= 1)
        self.bn10 = nn.BatchNorm2d(128)
        
        self.conv11 = nn.Conv2d(in_channels=128, out_channels= 128, kernel_size= 3)
        self.bn11 = nn.BatchNorm2d(128)
        
        self.conv12 = nn.Conv2d(in_channels=128, out_channels= 128, kernel_size= 3)
        self.bn12 = nn.BatchNorm2d(128)
        
        self.conv13 = nn.Conv2d(in_channels=128, out_channels= 160, kernel_size= 1)
        self.bn13 = nn.BatchNorm2d(160)
        
        self.conv14 = nn.Conv2d(in_channels=160, out_channels= 160, kernel_size= 1)
        self.bn14 = nn.BatchNorm2d(160)
        
        self.conv15 = nn.Conv2d(in_channels=160, out_channels= 160, kernel_size= 1)
        self.bn15 = nn.BatchNorm2d(160)
        
        self.conv16 = nn.Conv2d(in_channels=160, out_channels= 160, kernel_size= 1)
        self.bn16 = nn.BatchNorm2d(160)
        
        self.fully_connected = nn.Linear(in_features= 1440, out_features= 1)
        
    def forward(self, x):
        
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn3(self.conv3(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn4(self.conv4(out)), negative_slope = self.neg_slope)
        
        out = self.pooling(out)

        out = F.leaky_relu(self.bn5(self.conv5(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn6(self.conv6(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn7(self.conv7(out)), negative_slope = self.neg_slope)
        
        out = self.pooling(out)

        out = F.leaky_relu(self.bn8(self.conv8(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn9(self.conv9(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn10(self.conv10(out)), negative_slope = self.neg_slope)
        
        out = self.pooling(out)

        out = F.leaky_relu(self.bn11(self.conv11(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn12(self.conv12(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn13(self.conv13(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn14(self.conv14(out)), negative_slope = self.neg_slope)
        
        out = self.pooling(out)

        out = F.leaky_relu(self.bn15(self.conv15(out)), negative_slope = self.neg_slope)
        out = F.leaky_relu(self.bn16(self.conv16(out)), negative_slope = self.neg_slope)
        
        out = out.view((self.batch_size, 1440))
        out = self.fully_connected(out)
        
        return out
        
        