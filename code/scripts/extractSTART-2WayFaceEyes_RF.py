import numpy as np
import scipy.io as sio
import os, cv2, time, datetime, random
from copy import deepcopy

from insightface.app import FaceAnalysis


def squareEyeBBox(coords):
	[minX, minY] = coords.min(axis=0)
	[maxX, maxY] = coords.max(axis=0)
	centerX = (minX+maxX)/2
	centerY = (minY+maxY)/2
	s = max(maxX-minX, maxY-minY)
	
	# Eye Bounding Box --> twice in size than the square which tightly bounds the eye landmarks
	x = int(centerX - s)	# x = int(centerX - 2*s/2)
	y = int(centerY - s)	# y = int(centerY - 2*s/2)
	w = h = 2*s
	return [x, y, w, h]


def getFrameName(folderNum, frameNum):
	if (folderNum in ['1', '2', '3', '7', '8']):
		return 'frame%d.jpg' % int(frameNum)
	elif (folderNum in ['4']):
		return '%d.png' % int(frameNum)
	elif (folderNum in ['5', '6', '9', '10', '11']):
		if (int(frameNum) < 1000):
			return '%03d.bmp' % int(frameNum)
		else:
			return '%d.bmp' % int(frameNum)
	else:
		print("ERROR: 'folderNum' expected in range [1,11], but got %d" % folderNum)
		exit()


def prepare_START_2Way_RF_dataset():
	with open(folderNum_path, 'r') as folderNum_file:
		folderNums = folderNum_file.read().splitlines()
	with open(frameNum_path, 'r') as frameNum_file:
		frameNums = frameNum_file.read().splitlines()
	with open(gazeLabel_path, 'r') as gazeLabel_file:
		gazeLabels = gazeLabel_file.read().splitlines()

	metapath = os.path.join(out_data_dir, meta_filename)
	metadata = {
		'labelRecNum': [],
		'frameIndex': [],
		'gazeLR': [],
		'labelFaceGrid': [],
		'labelTrain': [],
		'labelVal': [],
		'labelTest': [],
		'invalid_gazeLabel_recNum_frameId': [],
		'0Faces_recNum_frameId': [],
		'gt1Faces_recNum_frameId': [],
		'faceSzGTframeSz_recNum_frameId': []
	}

	app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
	app.prepare(ctx_id=0, det_size=(640, 640))

	frameSizes = []		# list which stores list (i.e. [cols, rows]) for each valid frame sizes
	faceBboxes = []		# list which stores np array (i.e. array([X, Y, W, H]))for each valid face
	F = len(folderNums)
	tic = time.time()
	for f, (folderNum, frameNum, gazeLabel) in enumerate(zip(folderNums, frameNums, gazeLabels)):
		if (gazeLabel not in ['L', 'R']):
			metadata['invalid_gazeLabel_recNum_frameId'].append([int(folderNum), int(frameNum)])
			continue

		framePath = os.path.join(in_data_dir, child_dirs[folderNum], getFrameName(folderNum, frameNum))
		frame = cv2.imread(framePath)

		faces = app.get(frame)
		if (len(faces) == 0):
			metadata['0Faces_recNum_frameId'].append([int(folderNum), int(frameNum)])
		elif (len(faces) > 1):
			metadata['gt1Faces_recNum_frameId'].append([int(folderNum), int(frameNum)])
		else:
			face = faces[0]
			x1, y1, x2, y2 = face.bbox[0], face.bbox[1], face.bbox[2], face.bbox[3]
			w, h = x2-x1+1, y2-y1+1

			frameW, frameH = frame.shape[1], frame.shape[0]
			frameSizes.append([frameW, frameH])		# append [cols, rows] of frame
			faceBboxes.append([x1, y1, w, h])									# append [X, Y, W, H] of detected face

			# Square face crop centered on the center of RF pred face BBox: just increase the width/height equally from both sides
			if (w <= h):
				ofst = (h-w)//2
				x1 = x1-ofst
				w = h
			else:
				ofst = (w-h)//2
				y1 = y1-ofst
				h = w
			# If adjusted square face's width/height > frame's width/height, skip the frame
			if (w > frameW) or (h > frameH):
				metadata['faceSzGTframeSz_recNum_frameId'].append([int(folderNum), int(frameNum)])
				frameSizes.pop()
				faceBboxes.pop()
				continue
			# If adjusted square face is outside the frame, shift it accordingly
			x1 = 0 if x1 < 0 else x1
			y1 = 0 if y1 < 0 else y1
			x1 = frameW-w if x1+w > frameW else x1
			y1 = frameH-h if y1+h > frameH else y1
			faceCrop = frame[int(y1):int(y1+h), int(x1):int(x1+w), :]

			# Square eye crops (Subject's Left eye 2D lmks [87:97], Subject's Right eye 2D lmks [33:43]): twice in size than the square which tightly bounds the eye landmarks
			lmks_2d = face.landmark_2d_106
			leftEyeLM = lmks_2d[87:97]
			rightEyeLM = lmks_2d[33:43]
			leftEye_x1, leftEye_y1, leftEye_w, leftEye_h = squareEyeBBox(leftEyeLM)
			leftEye_x2 = leftEye_x1 + leftEye_w
			leftEye_y2 = leftEye_y1 + leftEye_h
			rightEye_x1, rightEye_y1, rightEye_w, rightEye_h = squareEyeBBox(rightEyeLM)
			rightEye_x2 = rightEye_x1 + rightEye_w
			rightEye_y2 = rightEye_y1 + rightEye_h

			leftEye_x1 = 0 if leftEye_x1 < 0 else leftEye_x1
			leftEye_y1 = 0 if leftEye_y1 < 0 else leftEye_y1
			leftEye_x2 = frameW if leftEye_x2 > frameW else leftEye_x2
			leftEye_y2 = frameH if leftEye_y2 > frameH else leftEye_y2
			rightEye_x1 = 0 if rightEye_x1 < 0 else rightEye_x1
			rightEye_y1 = 0 if rightEye_y1 < 0 else rightEye_y1
			rightEye_x2 = frameW if rightEye_x2 > frameW else rightEye_x2
			rightEye_y2 = frameH if rightEye_y2 > frameH else rightEye_y2
			leftEyeCrop = frame[int(leftEye_y1):int(leftEye_y2), int(leftEye_x1):int(leftEye_x2), :]
			rightEyeCrop = frame[int(rightEye_y1):int(rightEye_y2), int(rightEye_x1):int(rightEye_x2), :]

			faceDir = os.path.join(out_data_dir, '%05d' % int(folderNum), 'appleFace')
			leftEyeDir = os.path.join(out_data_dir, '%05d' % int(folderNum), 'appleLeftEye')
			rightEyeDir = os.path.join(out_data_dir, '%05d' % int(folderNum), 'appleRightEye')
			if (not os.path.exists(faceDir)):
				os.makedirs(faceDir, 0o777)
			if (not os.path.exists(leftEyeDir)):
				os.makedirs(leftEyeDir, 0o777)
			if (not os.path.exists(rightEyeDir)):
				os.makedirs(rightEyeDir, 0o777)

			cv2.imwrite(os.path.join(faceDir, '%05d.jpg' % int(frameNum)), faceCrop)
			cv2.imwrite(os.path.join(leftEyeDir, '%05d.jpg' % int(frameNum)), leftEyeCrop)
			cv2.imwrite(os.path.join(rightEyeDir, '%05d.jpg' % int(frameNum)), rightEyeCrop)

			metadata['labelRecNum'].append(int(folderNum))
			metadata['frameIndex'].append(int(frameNum))
			metadata['gazeLR'].append(0 if gazeLabel == 'L' else 1)

			if (f % 100 == 0):
				toc = time.time()
				
				frameSizesArr = np.array(frameSizes)
				pixPerGridArr = frameSizesArr / FACE_GRID_SIZE
				faceBboxesArr = np.array(faceBboxes)

				metadata['labelFaceGrid'] = np.zeros(faceBboxesArr.shape, dtype=np.int8)
				metadata['labelFaceGrid'][:,0:2] = np.around(faceBboxesArr[:,0:2] / pixPerGridArr)	# store [gridX1, gridY1]
				metadata['labelFaceGrid'][:,2:4] = np.around((faceBboxesArr[:,0:2] + faceBboxesArr[:,2:4]) / pixPerGridArr) - 1	# store [gridX2, gridY2]
				metadata['labelFaceGrid'][:,2:4] -= metadata['labelFaceGrid'][:,0:2] - 1	# store [gridX, gridY, gridW, gridH]

				metadata_copy = deepcopy(metadata)
				metadata_copy['labelRecNum'] = np.array(metadata_copy['labelRecNum'], dtype=np.int16)
				metadata_copy['frameIndex'] = np.array(metadata_copy['frameIndex'], dtype=np.int32)
				metadata_copy['gazeLR'] = np.array(metadata_copy['gazeLR'], dtype=int)

				print('Processed %d/%d frames (%.2f' % (f, F, 100*f/F), '%)', ' |  Saved "%s"  |  Time = %s' % (metapath, datetime.timedelta(seconds=toc-tic)))
				sio.savemat(metapath, metadata_copy)

	frameSizesArr = np.array(frameSizes)
	pixPerGridArr = frameSizesArr / FACE_GRID_SIZE
	faceBboxesArr = np.array(faceBboxes)

	metadata['labelFaceGrid'] = np.zeros(faceBboxesArr.shape, dtype=np.int8)
	metadata['labelFaceGrid'][:,0:2] = np.around(faceBboxesArr[:,0:2] / pixPerGridArr)	# store [gridX1, gridY1]
	metadata['labelFaceGrid'][:,2:4] = np.around((faceBboxesArr[:,0:2] + faceBboxesArr[:,2:4]) / pixPerGridArr) - 1	# store [gridX2, gridY2]
	metadata['labelFaceGrid'][:,2:4] -= metadata['labelFaceGrid'][:,0:2] - 1	# store [gridX, gridY, gridW, gridH]

	metadata['labelRecNum'] = np.array(metadata['labelRecNum'], dtype=np.int16)
	metadata['frameIndex'] = np.array(metadata['frameIndex'], dtype=np.int32)
	metadata['gazeLR'] = np.array(metadata['gazeLR'], dtype=int)

	print('Processed %d/%d frames (%.2f' % (f, F, 100*f/F), '%)', ' |  Saved "%s"  |  Time = %s' % (metapath, datetime.timedelta(seconds=toc-tic)))
	sio.savemat(metapath, metadata)


def split_dataset(train_frac, val_frac):
	metapath = os.path.join(out_data_dir, meta_filename)
	metadata = sio.loadmat(metapath, squeeze_me=True, struct_as_record=False)
	
	F = len(metadata['labelRecNum'])
	train_F = int(train_frac * F)
	val_F = int(val_frac * F)
	test_F = F - (train_F + val_F)
	
	splits = [0]*train_F + [1]*val_F + [2]*test_F	# train=0, val=1, test=2
	random.shuffle(splits)
	splits = np.array(splits)
	
	train_idx = np.argwhere(splits == 0).flatten()
	val_idx = np.argwhere(splits == 1).flatten()
	test_idx = np.argwhere(splits == 2).flatten()

	labelTrain = np.zeros(F, dtype=np.bool)
	labelTrain[train_idx] = 1
	labelVal = np.zeros(F, dtype=np.bool)
	labelVal[val_idx] = 1
	labelTest = np.zeros(F, dtype=np.bool)
	labelTest[test_idx] = 1

	metadata['labelTrain'] = labelTrain
	metadata['labelVal'] = labelVal
	metadata['labelTest'] = labelTest
	sio.savemat(metapath, metadata)


if __name__ == '__main__':
	out_data_dir = 'data/RF_extracted_START-2Way_dataset'
	in_data_dir = 'data/csailDataCreator'
	folderNum_path = 'data/csailDataCreator/folderNum.txt'
	frameNum_path = 'data/csailDataCreator/frameNum.txt'
	gazeLabel_path = 'data/csailDataCreator/gazeLabel.txt'
	meta_filename = 'metadata.mat'
	FACE_GRID_SIZE = np.array([25, 25])		# [cols, rows]

	child_dirs = {
		'1':'child20',	# frame1.jpg - frameXYZ.jpg					-->  jpg_seq_gen()
		'2':'child28',	# frame1.jpg - frameXYZ.jpg					-->  jpg_seq_gen()
		'3':'child101',	# frame1.jpg - frameXYZ.jpg					-->  jpg_seq_gen()
		'4':'self',		# 1.png - XYZ.png							-->  png_seq_gen()
		'5':'child70',	# 000.bmp - 999.bmp ; 1000.bmp - 1224.bmp	-->  bmp_seq_gen()
		'6':'child72',	# 000.bmp - 999.bmp ; 1000.bmp - 1225.bmp	-->  bmp_seq_gen()
		'7':'child141',	# frame1.jpg - frameXYZ.jpg					-->  jpg_seq_gen()
		'8':'child162',	# frame1.jpg - frameXYZ.jpg					-->  jpg_seq_gen()
		'9':'kohli',	# 000.bmp - 999.bmp ; 1000.bmp - 1226.bmp	-->  bmp_seq_gen()
		'10':'nadir',	# 000.bmp - 999.bmp ; 1000.bmp - 1170.bmp	-->  bmp_seq_gen()
		'11':'yellowT'	# 000.bmp - 999.bmp ; 1000.bmp - 1227.bmp	-->  bmp_seq_gen()
	}

	if (not os.path.exists(out_data_dir)):
		os.makedirs(out_data_dir)

	# prepare_START_2Way_RF_dataset()

	# NOTE - train, val & test set to be divided after prepare_START_2Way_RF_dataset()
	train_frac = 0.7
	val_frac = 0.1
	# test --> remaining frames
	random.seed(0)
	split_dataset(train_frac, val_frac)
