import argparse
import os, sys
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable


'''
Pytorch model for the iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''


class ItrackerImageModel(nn.Module):
	# Used for both eyes (with shared weights) and the face (with unqiue weights)
	def __init__(self):
		super(ItrackerImageModel, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),				# output = Size([N, 96, 54, 54])
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),								# output = Size([N, 96, 26, 26])
			nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
			nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),	# output = Size([N, 256, 26, 26])
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),								# output = Size([N, 256, 12, 12])
			nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
			nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),			# output = Size([N, 384, 12, 12])
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),				# output = Size([N, 64, 12, 12])
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		x = self.features(x)		# output = Size([N, 64, 12, 12])
		x = x.view(x.size(0), -1)	# output = Size([N, 9216])
		return x

class FaceImageModel(nn.Module):
	
	def __init__(self):
		super(FaceImageModel, self).__init__()
		self.conv = ItrackerImageModel()
		self.fc = nn.Sequential(
			nn.Linear(12*12*64, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 64),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		x = self.conv(x)
		x = self.fc(x)
		return x

class FaceGridModel(nn.Module):
	# Model for the face grid pathway
	def __init__(self, gridSize = 25):
		super(FaceGridModel, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(gridSize * gridSize, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, 128),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x



class ITrackerModel(nn.Module):


	def __init__(self, kind='regression'):
		super(ITrackerModel, self).__init__()
		self.eyeModel = ItrackerImageModel()	# output = Size([N, 9216])
		self.faceModel = FaceImageModel()		# output = Size([N, 64])
		self.gridModel = FaceGridModel()		# output = Size([N, 128])
		# Joining both eyes
		self.eyesFC = nn.Sequential(
			nn.Linear(2*12*12*64, 128),			# output = Size([N, 128])
			nn.ReLU(inplace=True),
			)
		# Joining everything
		if kind == 'regression':
			self.fc = nn.Sequential(
				nn.Linear(128+64+128, 128),			# output = Size([N, 128])
				nn.ReLU(inplace=True),
				nn.Linear(128, 2),					# output = Size([N, 2])
				)
		elif kind == 'classification':
			self.fc = nn.Sequential(
				nn.Linear(128+64+128, 128),			# output = Size([N, 128])
				nn.ReLU(inplace=True),
                nn.Linear(128, 2),
				# nn.Linear(128, 3),			# For L,R,C		# output = Size([N, 3])
				# nn.Linear(128, 4),		# For L,R,C,Out	# output = Size([N, 4])
				)
		else:
			print("INVALID kind of model (expected 'regression' or 'classification')..!!")
			sys.exit(1)

	def forward(self, faces, eyesLeft, eyesRight, faceGrids):
		# Eye nets
		xEyeL = self.eyeModel(eyesLeft)
		xEyeR = self.eyeModel(eyesRight)
		# Cat and FC
		xEyes = torch.cat((xEyeL, xEyeR), 1)
		xEyes = self.eyesFC(xEyes)

		# Face net
		xFace = self.faceModel(faces)
		xGrid = self.gridModel(faceGrids)

		# Cat all
		x = torch.cat((xEyes, xFace, xGrid), 1)
		x = self.fc(x)
		
		return x