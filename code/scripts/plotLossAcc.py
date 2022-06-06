import numpy as np
import torch, os
import matplotlib.pyplot as plt


CHECKPOINTS_PATH = 'saved_models/2WayGazeClassification/START-2Way/RF_subsetSTART-2Way-train-10percent_TL_checkpoints_train'
CHECKPOINT_LOAD_FILE = 'checkpoint_train_100.pth.tar'
val_epochs = list(range(25,101))
train_epochs = list(range(26,101))

GPU_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_checkpoint(filename=CHECKPOINT_LOAD_FILE):
	filename = os.path.join(CHECKPOINTS_PATH, filename)
	# print(filename)
	if not os.path.isfile(filename):
		return None
	state = torch.load(filename, map_location=GPU_device)
	return state


if __name__ == '__main__':
    saved = load_checkpoint()
    train_losses = saved['train_losses']
    val_accs = saved['val_accs']

    plt.figure()
    plt.plot(train_epochs, train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.savefig('results/plot_trainLossVSepoch.jpg')

    plt.figure()
    plt.plot(val_epochs, val_accs)
    plt.xlabel('Epochs')
    plt.ylabel('Val Accuracy')
    plt.savefig('results/plot_valAccuracyVSepoch.jpg')