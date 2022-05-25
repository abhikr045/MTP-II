'''
Train/test code for Classification.
'''

import math, shutil, os, time, argparse, csv, sys, datetime
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from ITrackerData import ITrackerData
from ITrackerModel import ITrackerModel

from collections import OrderedDict
from sklearn.utils.class_weight import compute_class_weight


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


# NOTE - check CHECKPOINTS_PATH before running
CHECKPOINTS_PATH = 'saved_models/2WayGazeClassification/RF_subsetUK-train-2.5percent_TL-noFace_checkpoints_train'
CHECKPOINT_LOAD_FILE = 'checkpoint_train_25.pth.tar'
CHECKPOINT_SAVE_FILE = 'checkpoint'
METAFILE = 'metadata_subset_train-2.5percent_gazeLR.mat'
MEAN_PATH = 'metadata/'

# NOTE - check which data to Train on and which data to Validate the accuracy on
TRAIN_ON = 'train'
VAL_ON = 'val'

TOT_VALID = 786732	# total valid frames in the dataset
KIND = 'classification'
LRC_device = 'Huawei'
MODEL_USED = 'DlibHoGMIT'
TABLET = 'HuaweiMediaPadM3Lite10'

# Making the code Reproducible
seedVal = 0
np.random.seed(seedVal)
torch.manual_seed(seedVal)
torch.cuda.manual_seed(seedVal)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
parser.add_argument('--data_path', help="Path to processed dataset. It should contain metadata.mat. Use prepareDataset.py.")
parser.add_argument('--sink', type=str2bool, nargs='?', const=True, default=False, help="Just sink and terminate.")
parser.add_argument('--reset', type=str2bool, nargs='?', const=True, default=False, help="Start from scratch (do not load).")
args = parser.parse_args()
# Different combinations of command-line arguments -
# sink	reset	-->	action
# T		T		-->	Accuracy testing without any training
# T		F		-->	only Testing
# F		T		-->	Training from starting
# F		F		-->	Resume Training from a checkpoint

# Change the flags to control what happens.
doLoad = not args.reset # Load checkpoint at the beginning
doTest = args.sink # Only run test, no training

workers = 4
epochs = 35
batch_size = 100 # Change if out of cuda memory

base_lr = 0.001
momentum = 0.9
weight_decay = 1e-4
prec1 = 0
best_prec1 = 0
lr = base_lr

GPU_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_checkpoint(filename=CHECKPOINT_LOAD_FILE):
	filename = os.path.join(CHECKPOINTS_PATH, filename)
	# print(filename)
	if not os.path.isfile(filename):
		return None
	state = torch.load(filename, map_location=GPU_device)
	return state


def save_checkpoint(state, is_best, filename=CHECKPOINT_SAVE_FILE):
	if not os.path.isdir(CHECKPOINTS_PATH):
		os.makedirs(CHECKPOINTS_PATH, 0o777)
	bestFilename = os.path.join(CHECKPOINTS_PATH, filename.replace('.pth.tar', '_best.pth.tar'))
	filename = os.path.join(CHECKPOINTS_PATH, filename)
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, bestFilename)


def adjust_learning_rate(epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = base_lr * (0.1 ** (epoch // 30))
	return lr
# 	for param_group in optimizer.state_dict()['param_groups']:
# 		param_group['lr'] = lr


def gazeXY2label(xy, tablet):
	if tablet == 'HuaweiMediaPadM3Lite10':
		minX, maxX, minY, maxY = -10.876, 10.876, -14.424, -0.83
	elif tablet == 'SamsungNote10SM_P600_P601':
		minX, maxX, minY, maxY = -8.77, 12.9, -14.4, -0.77
	else:
		print('Invalid tablet %s found..!!' % tablet)
		sys.exit(1)
	
	x = np.array(xy[:,0])
	y = np.array(xy[:,1])
	labels = np.zeros(len(xy), dtype=np.str_)
	xBounds = np.linspace(start=minX, stop=maxX, num=4)
	
	yOnScreen = (y >= minY) & (y <= maxY)
	left = (x >= xBounds[0]) & (x <= xBounds[1]) & yOnScreen
	center = (x > xBounds[1]) & (x < xBounds[2]) & yOnScreen
	right = (x >= xBounds[2]) & (x <= xBounds[3]) & yOnScreen

	labels[left] = 'L'
	labels[center] = 'C'
	labels[right] = 'R'

	return labels


def classifAccuracy(pred, truth):
	correct = pred == truth
	Nr = torch.sum(correct).item()
	Dr = len(pred)
	return 100*Nr/Dr


accOnlyLRC = torch.empty((0,2), dtype=torch.long)
def main():
	global args, best_prec1, weight_decay, momentum, accOnlyLRC

	# To load a regression model (i.e. output = XY coordinates) first & then use it for Transfer Learning for classification: kind = 'regression'
	# To load OR train from scratch a classification model (i.e. output = gazeLRC): kind = 'classification'
	model = ITrackerModel(kind=KIND)
	# model = torch.nn.DataParallel(model)
	model.to(device=GPU_device)
	imSize = (224,224)
	# cudnn.benchmark = True   

	epoch = 1
	if doLoad:
		saved = load_checkpoint()
		if saved:
			print('Loading checkpoint of epoch %05d with best_prec1 %.5f (which is the mean L2 (i.e. linear) error in cm)...' % (saved['epoch'], saved['best_prec1']))
			state = saved['state_dict']

			# Removing 'faceModel' from the pre-trained model (trained with 'faceModel' in Phase-1 of TL) to be loaded for Phase-2 of TL without face input
			modulesToRemove = []
			for k in state:
				if ('faceModel' in k):
					modulesToRemove.append(k)
			for moduleToRemove in modulesToRemove:
				del state[moduleToRemove]
			# Since face input is removed from NN, no. of input neurons in linear layer 'fc.0' changes from 320 to 256. So, 'fc.0.weight' needs to be of shape (128,256), instead of (128,320) saved in the model to be loaded.
			# So, temporarily loading dummy tensor of zeros of shape (128,256) in state['fc.0.weight']. Later, model.fc is initialized with new nn.Sequential(), so dummy tensor removed.
			state['fc.0.weight'] = torch.zeros((128, 256), device=GPU_device)

			try:
				model.module.load_state_dict(state)
			except:
				model.load_state_dict(state)
			epoch = saved['epoch'] + 1
			best_prec1 = saved['best_prec1']

			#############################
			##### Transfer Learning #####
			#############################
			# Required only if loading a regression model to be used for Transfer Learning for classification #
			#############################
			# Fix all conv layers
			# model.faceModel.conv.requires_grad = False
			model.eyeModel.requires_grad = False

			# Reset (i.e. trainable & initialized with random weights) last 2 FC layers with o/p of last FC layer as class labels L/R
			lin1_inFtrs = model.fc[0].in_features
			lin1_outFtrs = model.fc[0].out_features
			lin2_inFtrs = model.fc[2].in_features
			model.fc = nn.Sequential(
				nn.Linear(lin1_inFtrs, lin1_outFtrs),
				nn.ReLU(inplace=True),
				nn.Linear(lin2_inFtrs, 2),	# 2 outputs corresponding to LR
				)

			model.to(device=GPU_device)
			#############################
			##### Transfer Learning #####
			#############################
		else:
			print('Warning: Could not read checkpoint!')
			sys.exit(1)

	dataTrain = ITrackerData(dataPath=args.data_path, metafile=METAFILE, meanPath=MEAN_PATH, split=TRAIN_ON, imSize = imSize, kind=KIND, device=LRC_device)
	dataVal = ITrackerData(dataPath=args.data_path, metafile=METAFILE, meanPath=MEAN_PATH, split=VAL_ON, imSize = imSize, kind=KIND, device=LRC_device)

	train_loader = torch.utils.data.DataLoader(
		dataTrain,
		batch_size=batch_size, shuffle=True,
		num_workers=workers, worker_init_fn=lambda x: np.random.seed(seedVal), pin_memory=True)

	val_loader = torch.utils.data.DataLoader(
		dataVal,
		batch_size=batch_size, shuffle=False,
		num_workers=workers, worker_init_fn=lambda x: np.random.seed(seedVal), pin_memory=True)

	# Calculating class weights due to unbalanced training dataset
	##### DlibHoG_UK dataset #####
	# No. of gazeL in 'train' = 153,170
	# No. of gazeC in 'train' =  86,242
	# No. of gazeR in 'train' = 153,248
	##### DlibHoG_MIT dataset #####
	# No. of gazeL in 'train' =   174,249
	# No. of gazeC in 'train' =   466,161
	# No. of gazeR in 'train' =    47,217
	# No. of gazeOut in 'train' = 564,253

	# gazeLRC_classes = dataTrain.metadata['gazeLRC']
	# trainSplit = dataTrain.metadata['labelTrain']
	# trainGazeLRC = gazeLRC_classes[trainSplit == 1]
	# gazeClasses = np.array([0,1,2])	# For UK dataset OR iPadAir2 subset of MIT, 3 labels - L,C & R
	gazeLR_classes = dataTrain.metadata['gazeLR']
	trainSplit = dataTrain.metadata['labelTrain']
	trainGazeLR = gazeLR_classes[trainSplit == 1]
	gazeClasses = np.array([0,1])

	# gazeClasses = np.array([0,1,2,3])	# For MIT dataset, 4 labels - L,C,R & Out
	weights = compute_class_weight('balanced', classes=gazeClasses, y=trainGazeLR)
	weights = weights / weights.sum()
	weights = torch.Tensor(weights)
	criterion = nn.CrossEntropyLoss(weight=weights).to(device=GPU_device)
	# criterion = classifAccuracy

	##### Specify the parameters to be optimized (i.e. only the trainable params) in Transfer Learning #####
	trainableParams = list(model.eyesFC[0].parameters()) + list(model.gridModel.parameters()) + list(model.fc.parameters())
	optimizer = torch.optim.SGD(trainableParams,
								base_lr, momentum=momentum,
								weight_decay=weight_decay)
	########################################################################################################
	# optimizer = torch.optim.SGD(model.parameters(),
	# 							base_lr, momentum=momentum,
	# 							weight_decay=weight_decay)

	# Quick test
	if doTest:
		validate(val_loader, model, criterion, epoch-1)

		# onlyLRCregion = {'pred_truth': accOnlyLRC.numpy()}
		# sio.savemat('../results/onlyLRCregionGaze-pred_truth.mat', onlyLRCregion)

		# onlyLRCid = accOnlyLRC[:,1] != 3
		# accOnlyLRC = accOnlyLRC[onlyLRCid]
		# print('Accuracy (considering frames only in LRC region of Huawei) = %s' % classifAccuracy(accOnlyLRC[:,0], accOnlyLRC[:,1]))

		################################################################################
		##### 1. Write 'predictedGaze.csv' in each rec dir
		##### 2. Output logs - 'recDir, accuracy' in ascending order of accuracy
		##### 3. Output logs - overall accuracy
		##### 4. Output logs - %age frames with gaze outside screen
		##### 5. Plot histogram of 'predX'
		################################################################################
		# predXY = calcGazaXY(val_loader, model, criterion, epoch-1)
		# assert predXY.shape == (TOT_VALID, 2), 'Shape of predXY NOT equal to (%d, 2) ..!!' % TOT_VALID
		# predLRC = gazeXY2label(predXY, tablet=TABLET)

		# recordings = os.listdir(args.data_path)
		# recordings = np.array(recordings, np.object)
		# recordings = recordings[[os.path.isdir(os.path.join(args.data_path, r)) for r in recordings]]
		# recordings.sort()
		# meta = loadMetadata(os.path.join(args.data_path, 'metadata.mat'))

		# recAcc = []
		# start = 0
		# for rec in recordings:
		# 	end = start + len(os.listdir(os.path.join(args.data_path, rec, 'appleFace')))
		# 	frames = meta['frameIndex'][start:end]
		# 	gazeLRC = meta['gazeLRC'][start:end]
		# 	info = np.stack((frames, gazeLRC), axis=1)
		# 	recXY = predXY[start:end]
		# 	recLRC = predLRC[start:end]
		# 	recAcc += [classifAccuracy(recLRC, gazeLRC)]

		# 	predGaze = np.concatenate((info, recXY), axis=1)
		# 	with open(os.path.join(args.data_path, rec, 'predictedGaze_meta-LRCviaTimestamps_%s.csv' % MODEL_USED), 'w') as predCSV:
		# 		predWriter = csv.writer(predCSV, delimiter=',')
		# 		predWriter.writerow(['frame', 'gazeLRC', 'predX', 'predY'])
		# 		predWriter.writerows(predGaze)
		# 	start = end

		# recAcc = np.array(recAcc)
		# sortOrder = np.argsort(recAcc)
		# print('Accuracy of each recording -')
		# print('Rec\t\tAccuracy(%)')
		# for indx in sortOrder:
		# 	print('%s\t%.2f' % (recordings[indx], recAcc[indx]))
		# totAcc = classifAccuracy(predLRC, meta['gazeLRC'])
		# print('Total Accuracy = %.2f %%' % totAcc)

		# out = predLRC == ''
		# recOut = meta['labelRecNum'][out]
		# framesOut = meta['frameIndex'][out]
		# print('\nrec_frame with predicted gazeXY outside screen -')
		# for o in range(len(recOut)):
		# 	print('%05d_%s' % (recOut[o], framesOut[o]))
		# print('Percentage of Gaze outside screen = %.2f %%' % (len(recOut)*100/len(meta['labelRecNum'])))

		# minX = int(min(predXY[:,0]))
		# maxX = int(max(predXY[:,0])) + 1
		# plt.hist(predXY[:,0], np.linspace(minX, maxX, num=maxX-minX+1), ec='black')
		# plt.xticks(ticks=range(5*int(minX/5), 5*(int(maxX/5)+1), 5))
		# plt.title('Histogram of Predicted gazeX coordinates')
		# plt.xlabel('Screen X (w.r.t camera)')
		# plt.ylabel('No. of frames')
		# plt.savefig(os.path.join('../results/hist_meta-LRCviaTimestamps_DlibHoGUK_%s.jpg' % MODEL_USED))
		################################################################################

		# Run below validation code in case program failed just after current epoch's training
		# prec1 = validate(val_loader, model, criterion, epoch)
		# print("Validation DONE. Overwriting checkpoint with validations's best_prec1 ...")
		# # remember best prec@1 and save checkpoint
		# is_best = prec1 < best_prec1
		# best_prec1 = min(prec1, best_prec1)
		# save_checkpoint({
		# 	'epoch': epoch,
		# 	'state_dict': model.state_dict(),
		# 	'best_prec1': best_prec1,
		# }, is_best, '%s_%s_%d.pth.tar' % (CHECKPOINT_SAVE_FILE, TRAIN_ON, epoch))
		# print('Checkpoint overwritten successfully.')
		
		return

	# for epoch in range(0, epoch):
	# 	adjust_learning_rate(optimizer, epoch)
	
	train_loss = []
	train_epochs = list(range(epoch, epochs+1))
	val_acc = []
	val_epochs = list(range(epoch-1, epochs+1))

	# evaluate on validation set
	prec1 = validate(val_loader, model, criterion, epoch-1)
	val_acc.append(prec1)
	best_prec1 = prec1
	print("Validation DONE. Saving checkpoint with validations's best_prec1 ...")
	# remember best prec@1 and save checkpoint
	save_checkpoint({
		'epoch': epoch-1,
		'state_dict': model.state_dict(),
		'best_prec1': best_prec1,
	}, False, '%s_%s_%d.pth.tar' % (CHECKPOINT_SAVE_FILE, TRAIN_ON, epoch-1))
	print('Checkpoint overwritten successfully.')

	#### Training Code for Classification #####
	for epoch in range(epoch, epochs+1):
		lr = adjust_learning_rate(epoch-1)
		for param_group in optimizer.state_dict()['param_groups']:
			param_group['lr'] = lr

		# train for one epoch
		cur_train_loss = train(train_loader, model, criterion, optimizer, epoch)
		train_loss.append(cur_train_loss)

		print("Training DONE. Not saving checkpoint for this ...")
		# save_checkpoint({
		# 	'epoch': epoch,
		# 	'state_dict': model.state_dict(),
		# 	'best_prec1': best_prec1,
		# 	'train_losses': train_loss,
		# 	'val_accs': val_acc
		# }, False, '%s_%s_%d.pth.tar' % (CHECKPOINT_SAVE_FILE, TRAIN_ON, epoch))
		# print('Checkpoint saved successfully.')
		# print('Training DONE. Not saving checkpoint for this...')

		# evaluate on validation set
		prec1 = validate(val_loader, model, criterion, epoch)
		val_acc.append(prec1)
		print("Validation DONE. Overwriting checkpoint with validations's best_prec1 ...")
		# remember best prec@1 and save checkpoint
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		if (is_best or epoch == epochs):
			save_checkpoint({
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'best_prec1': best_prec1,
				'train_losses': train_loss,
				'val_accs': val_acc
			}, is_best, '%s_%s_%d.pth.tar' % (CHECKPOINT_SAVE_FILE, TRAIN_ON, epoch))
			print('Checkpoint overwritten successfully.')
			# print('Validation DONE. Not saving checkpoint for this...')
	###########################################


def train(train_loader, model, criterion, optimizer, epoch):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()

	for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(train_loader):
		
		# measure data loading time
		data_time.update(time.time() - end)

		imFace = imFace.to(device=GPU_device, non_blocking=True)
		imEyeL = imEyeL.to(device=GPU_device, non_blocking=True)
		imEyeR = imEyeR.to(device=GPU_device, non_blocking=True)
		faceGrid = faceGrid.to(device=GPU_device, non_blocking=True)
		gaze = gaze.to(device=GPU_device, non_blocking=True)
		
		# Variable --> deprecated
		# imFace = torch.autograd.Variable(imFace, requires_grad = True)
		# imEyeL = torch.autograd.Variable(imEyeL, requires_grad = True)
		# imEyeR = torch.autograd.Variable(imEyeR, requires_grad = True)
		# faceGrid = torch.autograd.Variable(faceGrid, requires_grad = True)
		# gaze = torch.autograd.Variable(gaze, requires_grad = False)
		imFace.requires_grad_(True)
		imEyeL.requires_grad_(True)
		imEyeR.requires_grad_(True)
		faceGrid.requires_grad_(True)
		# gaze.requires_grad_(True)

		# compute output
		output = model(imFace, imEyeL, imEyeR, faceGrid)

		loss = criterion(output, gaze)
		
		losses.update(loss.data.item(), imFace.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		print('Epoch (train): [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
				   epoch, i+1, len(train_loader), batch_time=batch_time,
				   data_time=data_time, loss=losses))

	return losses.avg

def validate(val_loader, model, criterion, epoch):
	global accOnlyLRC
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	accuracies = AverageMeter()

	# switch to evaluate mode
	model.eval()
	end = time.time()

	oIndex = 0
	for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(val_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		imFace = imFace.to(device=GPU_device, non_blocking=True)
		imEyeL = imEyeL.to(device=GPU_device, non_blocking=True)
		imEyeR = imEyeR.to(device=GPU_device, non_blocking=True)
		faceGrid = faceGrid.to(device=GPU_device, non_blocking=True)
		gaze = gaze.to(device=GPU_device, non_blocking=True)
		
		# Variable --> deprecated
		# imFace = torch.autograd.Variable(imFace, requires_grad = False)
		# imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
		# imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
		# faceGrid = torch.autograd.Variable(faceGrid, requires_grad = False)
		# gaze = torch.autograd.Variable(gaze, requires_grad = False)
		imFace.requires_grad_(True)
		imEyeL.requires_grad_(True)
		imEyeR.requires_grad_(True)
		faceGrid.requires_grad_(True)
		# gaze.requires_grad_(True)

		# compute output
		with torch.no_grad():
			output = model(imFace, imEyeL, imEyeR, faceGrid)

		# accOnlyLRC = torch.cat((accOnlyLRC, torch.stack((torch.argmax(output, dim=1).cpu(), gaze.cpu()), dim=1)))

		loss = criterion(output, gaze)
		losses.update(loss.data.item(), imFace.size(0))

		# outputLRC = gazeXY2label(output.cpu().numpy(), tablet='HuaweiMediaPadM3Lite10')
		accuracy = classifAccuracy(torch.argmax(output, dim=1), gaze)
		accuracies.update(accuracy, imFace.size(0))
	 
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		print('Epoch (val): [{0}][{1}/{2}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'.format(
					epoch, i+1, len(val_loader), batch_time=batch_time,
					acc=accuracies))

	return accuracies.avg


def calcGazaXY(val_loader, model, criterion, epoch):
	batch_time = AverageMeter()
	data_time = AverageMeter()

	model.eval()
	xy = []
	end = time.time()
	for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze)  in enumerate(val_loader):
		data_time.update(time.time() - end)

		imFace = imFace.to(device=GPU_device, non_blocking=True)
		imEyeL = imEyeL.to(device=GPU_device, non_blocking=True)
		imEyeR = imEyeR.to(device=GPU_device, non_blocking=True)
		faceGrid = faceGrid.to(device=GPU_device, non_blocking=True)

		imFace.requires_grad_(True)
		imEyeL.requires_grad_(True)
		imEyeR.requires_grad_(True)
		faceGrid.requires_grad_(True)

		gaze = np.array(gaze)

		with torch.no_grad():
			output = model(imFace, imEyeL, imEyeR, faceGrid)
		
		output = output.cpu().numpy()
		xy += [output]

		batch_time.update(time.time() - end)
		end = time.time()

		# print('Epoch (val): [{0}][{1}/{2}]\t'
		# 			'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
		# 			'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
		# 			epoch, i+1, len(val_loader), batch_time=batch_time,
		# 			data_time=data_time))

	xy = np.concatenate(xy, axis=0)
	return xy


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


if __name__ == "__main__":
	tic = time.time()
	main()
	toc = time.time()
	print('Total processing time = %s' % datetime.timedelta(seconds = toc-tic))
	print('DONE')
