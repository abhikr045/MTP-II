import numpy as np
import cv2, os, time, datetime


def areAllFacesSquare():
	recNums = os.listdir(dataset_path)
	recNums = np.array(recNums, np.object)
	recNums = recNums[[os.path.isdir(os.path.join(dataset_path, r)) for r in recNums]]
	recNums.sort()

	cntFaces = 0
	diffsWH = []
	faces_diffWH = []
	diffsWH_gt1 = []
	faces_diffWH_gt1 = []
	for recNum in recNums:
		print('Processing %s . . .' % recNum)
		faceNames = os.listdir(os.path.join(dataset_path, recNum, 'appleFace'))
		for faceName in faceNames:
			facePath = os.path.join(dataset_path, recNum, 'appleFace', faceName)
			face = cv2.imread(facePath)
			h, w = face.shape[0], face.shape[1]
			cntFaces += 1

			if (h != w):
				diffsWH.append(abs(w-h))
				faces_diffWH.append([recNum, faceName])
				if (abs(w-h) > 1):
					diffsWH_gt1.append(abs(w-h))
					faces_diffWH_gt1.append([recNum, faceName])

	diffsWH = np.array(diffsWH)
	print('# rect faces = %d/%d (%.2f' % (len(diffsWH), cntFaces, 100*len(diffsWH)/cntFaces), ' %)')
	print('Max abs(w-h) = %d' % diffsWH.max())
	print('Mean abs(w-h) = %.2f' % diffsWH.mean())
	print('Std dev abs(w-h) = %.2f' % diffsWH.std())
	print('# faces with abs(w-h) > 1 = %d/%d (%.2f' % (len(diffsWH_gt1), cntFaces, 100*len(diffsWH_gt1)/cntFaces), ' %)')
	np.savez('results/rectFaces_RF_extracted_UK_dataset.npz', faces_diffWH=faces_diffWH, faces_diffWH_gt1=faces_diffWH_gt1)
	##### Result for DlibHoG_extracted_UK_dataset #####
	# #rect faces = 212471/786732 (27.01  %)
	# Max abs(w-h) = 36
	# Mean abs(w-h) = 1.01
	# Std dev abs(w-h) = 0.47
	# #faces with abs(w-h) > 1 = 149/786732 (0.02  %)

	##### Result for RF_extracted_UK_dataset #####
	# #rect faces = 186038/539508 (34.48  %)
	# Max abs(w-h) = 1
	# Mean abs(w-h) = 1.00
	# Std dev abs(w-h) = 0.00
	# #faces with abs(w-h) > 1 = 0/539508 (0.00  %)


if __name__ == '__main__' :
	dataset_path = 'data/RF_extracted_UK_dataset'

	tic = time.time()
	areAllFacesSquare()
	toc = time.time()
	print('Total processing time = %s' % datetime.timedelta(seconds=toc-tic))