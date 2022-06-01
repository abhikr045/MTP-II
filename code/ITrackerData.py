import torch.utils.data as data
import scipy.io as sio
from PIL import Image, ImageFilter
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re, sys

'''
Data loader for the iTracker.
Use prepareDataset.py to convert the dataset from http://gazecapture.csail.mit.edu/ to proper format.

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

def loadMetadata(filename, silent = False):
	try:
		# http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
		if not silent:
			print('\tReading metadata from %s...' % filename)
		metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
	except:
		print('\tFailed to read the meta file "%s"!' % filename)
		return None
	return metadata

class SubtractMean(object):
	"""Normalize a tensor image with mean.
	"""

	def __init__(self, meanImg):
		self.meanImg = transforms.ToTensor()(meanImg / 255)

	def __call__(self, tensor):
		"""
		Args:
			tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
		Returns:
			Tensor: Normalized image.
		"""       
		return tensor.sub(self.meanImg)

class blurFace(object):
	def __init__(self, r):
		self.r = r

	def __call__(self, PILimg):
		blurImg = PILimg.filter(ImageFilter.GaussianBlur(radius=self.r))
		# blurImg.save('results/blurFace/sampleBlurFace_r-%d.jpg' % self.r)
		return blurImg

		# blackImg = Image.new('RGB', (224,224), "black")
		# # blackImg.save('results/blackFace.jpg')
		# return blackImg

class ITrackerData(data.Dataset):
	def __init__(self, dataPath, metafile, meanPath, split='train', imSize=(224,224), gridSize=(25, 25), kind='regression', device=None):

		self.dataPath = dataPath
		self.imSize = imSize
		self.gridSize = gridSize
		self.kind = kind

		print('Loading iTracker dataset...')

		##### Check if metadata filename is correct #####
		metafilePath = os.path.join(dataPath, metafile)
		#################################################
		
		if metafilePath is None or not os.path.isfile(metafilePath):
			raise RuntimeError('There is no such file %s! Provide a valid dataset path.' % metafilePath)
		self.metadata = loadMetadata(metafilePath)
		if self.metadata is None:
			raise RuntimeError('Could not read metadata file %s! Provide a valid dataset path.' % metafilePath)

		if self.kind == 'classification' and 'MIT' in self.dataPath:
			if device is None or device not in ['Samsung', 'Huawei', 'iPadAir2', 'temp_HuaweiNoOutBounds']:
				raise RuntimeError("'device' parameter required (value = 'Samsung' or 'Huawei') in ITrackerData class objects for Classification on MIT dataset..!!")
			self.metadata['gazeLRC'] = self.metadata.pop('gazeLRC_%s' % device)

		# if device == 'iPadAir2':
		# 	meanFile = 'iPadAir2_mean'
		if 'MIT' in self.dataPath:
			meanFile = 'MIT_mean'
		elif 'UK' in self.dataPath:
			meanFile = 'UK_mean'
		else:
			raise RuntimeError('Unknown Mean files need to be loaded..!!')

		self.faceMean = loadMetadata(os.path.join(meanPath, '%s_face_224.mat' % meanFile))['image_mean']
		self.eyeLeftMean = loadMetadata(os.path.join(meanPath, '%s_left_224.mat' % meanFile))['image_mean']
		self.eyeRightMean = loadMetadata(os.path.join(meanPath, '%s_right_224.mat' % meanFile))['image_mean']
		
		self.transformFace = transforms.Compose([
            blurFace(r=15),
			transforms.Resize(self.imSize),
			transforms.ToTensor(),
			SubtractMean(meanImg=self.faceMean),
		])
		self.transformEyeL = transforms.Compose([
            # blurFace(r=15),
			transforms.Resize(self.imSize),
			transforms.ToTensor(),
			SubtractMean(meanImg=self.eyeLeftMean),
		])
		self.transformEyeR = transforms.Compose([
            # blurFace(r=15),
			transforms.Resize(self.imSize),
			transforms.ToTensor(),
			SubtractMean(meanImg=self.eyeRightMean),
		])


		if split == 'test':
			mask = self.metadata['labelTest']
		elif split == 'val':
			mask = self.metadata['labelVal']
		elif split == 'train':
			mask = self.metadata['labelTrain']
		else:
			print('ERROR: Invalid split = "%s" provided..!!' % split)
			exit(0)

		self.indices = np.argwhere(mask)[:,0]
		print('Loaded iTracker dataset split "%s" with %d records...' % (split, len(self.indices)))

	def loadImage(self, path):
		try:
			im = Image.open(path).convert('RGB')
		except OSError:
			raise RuntimeError('Could not read image: ' + path)
			#im = Image.new("RGB", self.imSize, "white")

		return im


	# NOTE - 'params' contains X,Y in 0-indexed coordinates
	def makeGrid(self, params):
		gridLen = self.gridSize[0] * self.gridSize[1]
		grid = np.zeros([gridLen,], np.float32)
		
		indsY = np.array([i // self.gridSize[1] for i in range(gridLen)])
		indsX = np.array([i % self.gridSize[1] for i in range(gridLen)])
		condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2]) 
		condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3]) 
		cond = np.logical_and(condX, condY)

		grid[cond] = 1
		return grid

	def __getitem__(self, index):
		# print('index = %d' % index, end=', ')
		index = self.indices[index]
		# print('img = %05d_%05d' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
		imFacePath = os.path.join(self.dataPath, '%05d/appleFace/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
		imEyeLPath = os.path.join(self.dataPath, '%05d/appleLeftEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
		imEyeRPath = os.path.join(self.dataPath, '%05d/appleRightEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))

		imFace = self.loadImage(imFacePath)
		imEyeL = self.loadImage(imEyeLPath)
		imEyeR = self.loadImage(imEyeRPath)

		imFace = self.transformFace(imFace)
		imEyeL = self.transformEyeL(imEyeL)
		imEyeR = self.transformEyeR(imEyeR)

		if self.kind == 'regression':
			##### For testing XY in [0,1] #####
			# gaze = np.array([self.metadata['labelDotXCam_0to1'][index], self.metadata['labelDotYCam_0to1'][index]], np.float32)
			###################################
			gaze = np.array([self.metadata['labelDotXCam'][index], self.metadata['labelDotYCam'][index]], np.float32)
			gaze = torch.FloatTensor(gaze)
		elif self.kind == 'classification':
			# gaze = self.metadata['gazeLR'][index]
			gaze = self.metadata['gazeLRC'][index]
		else:
			print("INVALID kind of dataset (expected 'regression' or 'classification')..!!")
			sys.exit(1)
		# gaze = self.metadata['gazeLR'][index]

		faceGrid = self.makeGrid(self.metadata['labelFaceGrid'][index,:])
		rec_frame = '%d_%d' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index])

		# to tensor
		faceGrid = torch.FloatTensor(faceGrid)

		return rec_frame, imFace, imEyeL, imEyeR, faceGrid, gaze
	
		
	def __len__(self):
		return len(self.indices)
