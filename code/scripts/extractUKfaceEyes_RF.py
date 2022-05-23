##### Preparing UK dataset with face/eyes extracted by RetinaFace #####
# To prepare this dataset, I'm using the following as starting point -
#   (a) UK_frames_dataset
#   (b) "DlibHoG_extracted_UK_dataset/metadata.mat" (since preparing the dataset from scratch is very difficult now after a 2 years gap with R&D work)

# "RF_extracted_UK_dataset/metadata.mat" will be written according to "DlibHoG_extracted_UK_dataset/metadata.mat" which contains contains -
# meta = {
#   'labelRecNum': [],      # NO CHANGE  -->  RecNum maps to folder no. in "UK_frames_dataset"
#   'frameIndex': [],       # NO CHANGE  -->  frameId maps to "UK_frame_dataset/RecNum/frames/frameId.jpg"
#   'gazeLRC': [],          # NO CHANGE
#   'labelFaceGrid': [],    # 2-D array: shape=(F,4) ; dtype=int8  -->  Face BBox as calculated by RetinaFace
#   'labelTrain': [],       # NO CHANGE
#   'labelVal': [],         # NO CHANGE
#   'labelTest': []         # NO CHANGE
# 	}

# "RF_extracted_UK_dataset/RecNum/apple{Face,LeftEye,RightEye}/frameId.jpg" will be written for only those frames present in "DlibHoG_extracted_UK_dataset/metadata.mat"


import numpy as np
import scipy.io as sio
import os, cv2, time, datetime

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

def prepare_UK_RF_dataset():
	DlibHoG_metadata = sio.loadmat(DlibHoG_metadata_path, squeeze_me=True, struct_as_record=False)
	rf_metadata = {
		'labelRecNum': [],
		'frameIndex': [],
		'gazeLRC': [],
		'labelFaceGrid': [],
		'labelTrain': [],
		'labelVal': [],
		'labelTest': [],
		'0Faces_recNum_frameId': [],
		'gt1Faces_recNum_frameId': [],
		'faceSzGTframeSz_recNum_frameId': []
	}

	frameSizes = []		# list which stores list (i.e. [cols, rows]) for each valid frame sizes
	faceBboxes = []		# list which stores np array (i.e. array([X, Y, W, H]))for each valid face

	app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
	app.prepare(ctx_id=0, det_size=(640, 640))

	F = len(DlibHoG_metadata['labelRecNum'])
	tic = time.time()
	for f in range(F):
		recNum = DlibHoG_metadata['labelRecNum'][f]
		frameId = DlibHoG_metadata['frameIndex'][f]
		framePath = os.path.join(dataset_path, '%05d'%recNum, 'frames', '%05d.jpg'%frameId)
		frame = cv2.imread(framePath)

		faces = app.get(frame)
		if (len(faces) == 0):
			rf_metadata['0Faces_recNum_frameId'].append([recNum, frameId])
		elif (len(faces) > 1):
			rf_metadata['gt1Faces_recNum_frameId'].append([recNum, frameId])
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
				rf_metadata['faceSzGTframeSz_recNum_frameId'].append([recNum, frameId])
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

			faceDir = os.path.join(rf_dataset_path, '%05d'%recNum, 'appleFace')
			leftEyeDir = os.path.join(rf_dataset_path, '%05d'%recNum, 'appleLeftEye')
			rightEyeDir = os.path.join(rf_dataset_path, '%05d'%recNum, 'appleRightEye')
			if (not os.path.exists(faceDir)):
				os.makedirs(faceDir, 0o777)
			if (not os.path.exists(leftEyeDir)):
				os.makedirs(leftEyeDir, 0o777)
			if (not os.path.exists(rightEyeDir)):
				os.makedirs(rightEyeDir, 0o777)

			cv2.imwrite(os.path.join(faceDir, '%05d.jpg'%frameId), faceCrop)
			cv2.imwrite(os.path.join(leftEyeDir, '%05d.jpg'%frameId), leftEyeCrop)
			cv2.imwrite(os.path.join(rightEyeDir, '%05d.jpg'%frameId), rightEyeCrop)

			rf_metadata['labelRecNum'].append(recNum)
			rf_metadata['frameIndex'].append(frameId)
			rf_metadata['gazeLRC'].append(DlibHoG_metadata['gazeLRC'][f])
			rf_metadata['labelTrain'].append(DlibHoG_metadata['labelTrain'][f])
			rf_metadata['labelVal'].append(DlibHoG_metadata['labelVal'][f])
			rf_metadata['labelTest'].append(DlibHoG_metadata['labelTest'][f])

			if (f % 100 == 0):
				toc = time.time()
				
				frameSizesArr = np.array(frameSizes)
				pixPerGridArr = frameSizesArr / FACE_GRID_SIZE
				faceBboxesArr = np.array(faceBboxes)

				rf_metadata['labelFaceGrid'] = np.zeros(faceBboxesArr.shape, dtype=np.int8)
				rf_metadata['labelFaceGrid'][:,0:2] = np.around(faceBboxesArr[:,0:2] / pixPerGridArr)	# store [gridX1, gridY1]
				rf_metadata['labelFaceGrid'][:,2:4] = np.around((faceBboxesArr[:,0:2] + faceBboxesArr[:,2:4]) / pixPerGridArr) - 1	# store [gridX2, gridY2]
				rf_metadata['labelFaceGrid'][:,2:4] -= rf_metadata['labelFaceGrid'][:,0:2] - 1	# store [gridX, gridY, gridW, gridH]

				print('Processed %d/%d frames (%.2f' % (f, F, 100*f/F), '%)', ' |  Saved "%s"  |  Time = %s' % (rf_metadata_path, datetime.timedelta(seconds=toc-tic)))
				sio.savemat(rf_metadata_path, rf_metadata)

	frameSizesArr = np.array(frameSizes)
	pixPerGridArr = frameSizesArr / FACE_GRID_SIZE
	faceBboxesArr = np.array(faceBboxes)

	rf_metadata['labelFaceGrid'] = np.zeros(faceBboxesArr.shape, dtype=np.int8)
	rf_metadata['labelFaceGrid'][:,0:2] = np.around(faceBboxesArr[:,0:2] / pixPerGridArr)	# store [gridX1, gridY1]
	rf_metadata['labelFaceGrid'][:,2:4] = np.around((faceBboxesArr[:,0:2] + faceBboxesArr[:,2:4]) / pixPerGridArr) - 1	# store [gridX2, gridY2]
	rf_metadata['labelFaceGrid'][:,2:4] -= rf_metadata['labelFaceGrid'][:,0:2] - 1	# store [gridX, gridY, gridW, gridH]

	print('Writing out the metadata.mat to %s...' % rf_metadata_path)
	sio.savemat(rf_metadata_path, rf_metadata)


if __name__ == '__main__':
	dataset_path = 'data/UK_frames_dataset'
	DlibHoG_metadata_path = 'data/DlibHoG_extracted_UK_dataset/metadata.mat'
	rf_dataset_path = 'data/RF_extracted_UK_dataset'
	rf_metadata_path = os.path.join(rf_dataset_path, 'metadata.mat')
	FACE_GRID_SIZE = np.array([25, 25])		# [cols, rows]

	if (not os.path.exists(rf_dataset_path)):
		os.makedirs(rf_dataset_path, 0o777)

	prepare_UK_RF_dataset()