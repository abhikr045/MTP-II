import numpy as np
import scipy.io as sio
import os, time, datetime


if __name__ == '__main__':
    metapath_gazeLRC = 'data/RF_extracted_UK_dataset/metadata_subset_train-2.5percent.mat'
    metapath_gazeLR = 'data/RF_extracted_UK_dataset/metadata_subset_train-2.5percent_gazeLR.mat'
    metapath_gazeC = 'data/RF_extracted_UK_dataset/metadata_subset_train-2.5percent_gazeC.mat'

    meta_gazeLRC = sio.loadmat(metapath_gazeLRC, squeeze_me=True, struct_as_record=False)

    id_gazeLR = np.argwhere(meta_gazeLRC['gazeLRC'] != 1).flatten()
    id_gazeC = np.argwhere(meta_gazeLRC['gazeLRC'] == 1).flatten()

    meta_gazeLR = {
        'labelRecNum': meta_gazeLRC['labelRecNum'][id_gazeLR],
        'frameIndex': meta_gazeLRC['frameIndex'][id_gazeLR],
        'gazeLR': meta_gazeLRC['gazeLRC'][id_gazeLR],
        'labelFaceGrid': meta_gazeLRC['labelFaceGrid'][id_gazeLR],
        'labelTrain': meta_gazeLRC['labelTrain'][id_gazeLR],
        'labelVal': meta_gazeLRC['labelVal'][id_gazeLR],
        'labelTest': meta_gazeLRC['labelTest'][id_gazeLR]
    }
    # change gaze labels for L & R from 0 & 2 to 0 & 1, respectively
    idR_gazeLR = np.argwhere(meta_gazeLR['gazeLR'] == 2)
    meta_gazeLR['gazeLR'][idR_gazeLR] = 1

    meta_gazeC = {
        'labelRecNum': meta_gazeLRC['labelRecNum'][id_gazeC],
        'frameIndex': meta_gazeLRC['frameIndex'][id_gazeC],
        'gazeC': np.zeros(len(id_gazeC), dtype=np.int8),    # gaze label for C in metadata_gazeC = 0
        'labelFaceGrid': meta_gazeLRC['labelFaceGrid'][id_gazeC],
        'labelTrain': np.zeros(len(id_gazeC), dtype=bool),
        'labelVal': np.zeros(len(id_gazeC), dtype=bool),
        'labelTest': np.ones(len(id_gazeC), dtype=bool)  # split of all gazeC data = 'test'
    }

    sio.savemat(metapath_gazeLR, meta_gazeLR)
    sio.savemat(metapath_gazeC, meta_gazeC)