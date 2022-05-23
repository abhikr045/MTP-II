import numpy as np
import scipy.io as sio


def prepSubsetMetadata():
	metadata = sio.loadmat(metapath, squeeze_me=True, struct_as_record=False)
	trainIds = np.argwhere(metadata['labelTrain'])[:,0]
	valIds = np.argwhere(metadata['labelVal'])[:,0]
	testIds = np.argwhere(metadata['labelTest'])[:,0]
	
	train_sz = len(trainIds)
	subset_train_sz = int(subset_trainPercent*train_sz/100)
	np.random.shuffle(trainIds)
	subset_trainIds = trainIds[:subset_train_sz]
	ids_in_subset = np.concatenate((subset_trainIds, valIds, testIds))

	subset_metadata = {
	    'labelRecNum': metadata['labelRecNum'][ids_in_subset],
	    'frameIndex': metadata['frameIndex'][ids_in_subset],
	    'gazeLRC': metadata['gazeLRC'][ids_in_subset],
	    'labelFaceGrid': metadata['labelFaceGrid'][ids_in_subset],
	    'labelTrain': metadata['labelTrain'][ids_in_subset],
	    'labelVal': metadata['labelVal'][ids_in_subset],
	    'labelTest': metadata['labelTest'][ids_in_subset],
	}

	sio.savemat(subset_metapath, subset_metadata)

if __name__ == '__main__':
	metapath = 'data/RF_extracted_UK_dataset/metadata.mat'
	# 100 batches/epoch (batch_sz=100) = 6 mins/epoch  ==>  1 hr for 10 epochs  ==>  10000 samples / 391265 train samples = 2.5% of train set (considering only train time; i.e., val time not included)
	subset_trainPercent = 2.5
	subset_metapath = 'data/RF_extracted_UK_dataset/metadata_subset_train-%.1fpercent.mat' % subset_trainPercent
	np.random.seed(0)

	prepSubsetMetadata()