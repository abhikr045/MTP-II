import numpy as np
import os, csv

if __name__ == '__main__':
	outGazeLR_CSV = open('results/RF_START-2Way_postClassif-noFace_outGazeLR.csv', 'r')
	outGazeLR_reader = csv.reader(outGazeLR_CSV)
	next(outGazeLR_reader)

	subj_correct_total = {1:[0,0], 2:[0,0], 3:[0,0], 4:[0,0], 5:[0,0], 6:[0,0], 7:[0,0], 8:[0,0], 9:[0,0], 10:[0,0], 11:[0,0]}
	for r, row in enumerate(outGazeLR_reader):
		recNum = int(row[0])
		grTr = int(row[2])
		pred = int(row[3])

		if (grTr == pred):
			subj_correct_total[recNum][0] += 1
		subj_correct_total[recNum][1] += 1

	# Final train/val/test splits
	train = [2,3,4,9,10,6]
	val = [5,7]
	test = [1,8,11]

	train_correct = 0
	train_total = 0
	print('##### train #####')
	for subj in train:
		print('[train] Subj-%d : %d/%d correct = %.2f' % (subj, subj_correct_total[subj][0], subj_correct_total[subj][1], 100 * subj_correct_total[subj][0] / subj_correct_total[subj][1]), '%',  ' accuracy')
		train_correct += subj_correct_total[subj][0]
		train_total += subj_correct_total[subj][1]
	print('[train] : %d/%d correct = %.2f' % (train_correct, train_total, 100 * train_correct / train_total), '%', ' accuracy' )

	val_correct = 0
	val_total = 0
	print('##### val #####')
	for subj in val:
		print('[val] Subj-%d : %d/%d correct = %.2f' % (subj, subj_correct_total[subj][0], subj_correct_total[subj][1], 100 * subj_correct_total[subj][0] / subj_correct_total[subj][1]), '%',  ' accuracy')
		val_correct += subj_correct_total[subj][0]
		val_total += subj_correct_total[subj][1]
	print('[val] : %d/%d correct = %.2f' % (val_correct, val_total, 100 * val_correct / val_total), '%', ' accuracy' )

	test_correct = 0
	test_total = 0
	print('##### test #####')
	for subj in test:
		print('[test] Subj-%d : %d/%d correct = %.2f' % (subj, subj_correct_total[subj][0], subj_correct_total[subj][1], 100 * subj_correct_total[subj][0] / subj_correct_total[subj][1]), '%',  ' accuracy')
		test_correct += subj_correct_total[subj][0]
		test_total += subj_correct_total[subj][1]
	print('[test] : %d/%d correct = %.2f' % (test_correct, test_total, 100 * test_correct / test_total), '%', ' accuracy' )

	outGazeLR_CSV.close()