import numpy as np
import matplotlib.pyplot as plt
import csv

if __name__ == '__main__':
    mode = 'test'
    probC_window_test = 0.7
    outProbCSV = open('results/RF_subset2WayUK_TL-blurFace-r15_outProb-allTest.csv', 'r')
    accPlot_filename = 'results/RF_subset2WayUK_TL-blurFace-r15_accuracy-allTest.jpg'
    
    outProbReader = csv.reader(outProbCSV)
    next(outProbReader)

    grTr3WayGaze_list = []
    probL_list = []
    probR_list = []

    for f, row in enumerate(outProbReader):
        grTr3WayGaze = int(float(row[0]))
        probL = float(row[1])
        probR = float(row[2])
        assert abs(1 - (probL+probR)) < 0.001, "ERROR: outProb_L + outProb_R != 1"

        grTr3WayGaze_list.append(grTr3WayGaze)
        probL_list.append(probL)
        probR_list.append(probR)

    probC_window = []
    gazeC_accs = []
    gazeLRC_accs = []
    gazeLR_accs = []
    maxLRCacc = float('-inf')
    maxLRCacc_probCwindow = 0

    if (mode == 'val'):
        range_probC_halfWindow = np.arange(0.025, 0.5, 0.025)
    elif (mode == 'test'):
        range_probC_halfWindow = np.array([probC_window_test/2])
    else:
        print("ERROR: Invalid mode '%s'" % mode)
        exit()

    for probC_halfWindow in range_probC_halfWindow:
        prob_low = 0.5 - probC_halfWindow
        prob_high = 0.5 + probC_halfWindow

        confusionMtx = np.zeros((3,3))
        for f, (grTr3WayGaze, probL, probR) in enumerate(zip(grTr3WayGaze_list, probL_list, probR_list)):
            if (probL < prob_low) and (probR > prob_high):
                confusionMtx[grTr3WayGaze, 2] += 1
            elif (probL > prob_high) and (probR < prob_low):
                confusionMtx[grTr3WayGaze, 0] += 1
            elif (probL >= prob_low) and (probL <= prob_high) and (probR >= prob_low) and (probR <= prob_high):
                confusionMtx[grTr3WayGaze, 1] += 1

        gazeC_acc = 100*confusionMtx[1,1]/np.sum(confusionMtx[1])
        gazeLRC_acc = 100*(np.trace(confusionMtx))/np.sum(confusionMtx)
        gazeLR_acc = 100*(confusionMtx[0,0] + confusionMtx[2,2])/(np.sum(confusionMtx[0]) + np.sum(confusionMtx[2]))

        if (gazeLRC_acc > maxLRCacc):
            maxLRCacc = gazeLRC_acc
            maxLRCacc_probCwindow = prob_high - prob_low

        probC_window.append(prob_high - prob_low)
        gazeC_accs.append(gazeC_acc)
        gazeLRC_accs.append(gazeLRC_acc)
        gazeLR_accs.append(gazeLR_acc)

    if (mode == 'val'):
        plt.plot(probC_window, gazeC_accs, 'g')
        plt.plot(probC_window, gazeLRC_accs, 'b')
        plt.plot(probC_window, gazeLR_accs, 'r')
        plt.xlabel('Prob(M) window from 0.5')
        plt.ylabel('Accuracies (%)')
        plt.xticks(probC_window[1::2])
        plt.legend(['Accuracy (M)', 'Accuracy(LMR)', 'Accuracy(LR)'])
        plt.vlines(x=probC_window, ymin=0.0, ymax=100.0, color='gray', linestyle='dotted')
        plt.vlines(x=maxLRCacc_probCwindow, ymin=0.0, ymax=100.0, color='gray', linestyle='dashed')
        plt.savefig(accPlot_filename)

        print("Max LRC accuracy = %.3f" % maxLRCacc)
        print("Max LRC accuracy occurs at prob(C) window = %.3f" % maxLRCacc_probCwindow)

        acc_table = np.array([probC_window, gazeC_accs, gazeLRC_accs, gazeLR_accs]).T
        print(acc_table)
    elif (mode == 'test'):
        print("Prob(C) window = %.3f" % probC_window[0])
        print("Accuracy (C) = %.3f" % gazeC_accs[0])
        print("Accuracy (LRC) = %.3f" % gazeLRC_accs[0])
        print("Accuracy (LR) = %.3f" % gazeLR_accs[0])
    else:
        print("ERROR: Invalid mode '%s'" % mode)
        exit()