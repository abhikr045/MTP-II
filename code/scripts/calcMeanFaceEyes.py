import numpy as np
import scipy.io as sio
import os, cv2, time, datetime


def calcMeanFaceEyes():
	meta = sio.loadmat(metaPath, squeeze_me=True, struct_as_record=False)

	sumFace = np.zeros((224, 224, 3), dtype=np.float32)
	sumEyeL = np.zeros((224, 224, 3), dtype=np.float32)
	sumEyeR = np.zeros((224, 224, 3), dtype=np.float32)

	trainID = np.argwhere(meta['labelTrain'] == 1).flatten()
	T = len(trainID)
	metaRecNum = meta['labelRecNum']
	metaFrameNum = meta['frameIndex']

	tic = time.time()
	for t, Id in enumerate(trainID):
		facePath = os.path.join(dataPath, '%05d' % metaRecNum[Id], 'appleFace', '%05d.jpg' % metaFrameNum[Id])
		leftEyePath = os.path.join(dataPath, '%05d' % metaRecNum[Id], 'appleLeftEye', '%05d.jpg' % metaFrameNum[Id])
		rightEyePath = os.path.join(dataPath, '%05d' % metaRecNum[Id], 'appleRightEye', '%05d.jpg' % metaFrameNum[Id])

		face = cv2.resize(cv2.imread(facePath).astype(np.float32), (224,224))
		leftEye = cv2.resize(cv2.imread(leftEyePath).astype(np.float32), (224,224))
		rightEye = cv2.resize(cv2.imread(rightEyePath).astype(np.float32), (224,224))

		if (t % 100 == 0):
			toc = time.time()
			print('Processed %d/%d training frames (%.2f' % (t, T, 100*t/T), '%)', ' |  Time = %s' % datetime.timedelta(seconds=toc-tic))

		sumFace += face
		sumEyeL += leftEye
		sumEyeR += rightEye

	meanFace = sumFace / T
	meanEyeL = sumEyeL / T
	meanEyeR = sumEyeR / T

	meanFaceMat = {'image_mean': meanFace}
	meanEyeLMat = {'image_mean': meanEyeL}
	meanEyeRMat = {'image_mean': meanEyeR}

	print('Total training Frames = %d' % T)
	sio.savemat('metadata/%s_mean_face_224.mat' % mean_base_filename, meanFaceMat)
	sio.savemat('metadata/%s_mean_left_224.mat' % mean_base_filename, meanEyeLMat)
	sio.savemat('metadata/%s_mean_right_224.mat' % mean_base_filename, meanEyeRMat)


if __name__ == '__main__':
	dataPath = 'data/RF_extracted_START-2Way_dataset'
	metaPath = os.path.join(dataPath, 'metadata.mat')
	mean_base_filename = 'START-2Way'

	calcMeanFaceEyes()