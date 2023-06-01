import os
import sys
import numpy as np
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score

NPY_FILE_PATH = "../npy_files/27May"

filenames = os.listdir(path=NPY_FILE_PATH)
filenames = sorted(filenames, key=lambda x: x.split('_')[-1])

def hamming_score(y_true, y_pred):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    
    return np.mean(acc_list)

file_count = 0
while file_count < len(filenames):
        run_name = filenames[file_count].split('.')[0].split('_')[-1]
        if 'labels' in filenames[file_count]:
                labels_data = np.load(f'{NPY_FILE_PATH}/{filenames[file_count]}', allow_pickle=True)
                preds_data = np.load(f'{NPY_FILE_PATH}/{filenames[file_count+1]}', allow_pickle=True)
        else:
                labels_data = np.load(f'{NPY_FILE_PATH}/{filenames[file_count+1]}', allow_pickle=True)
                preds_data = np.load(f'{NPY_FILE_PATH}/{filenames[file_count]}', allow_pickle=True)
        labels = []
        preds = []
        for i in range(len(labels_data)):
                for j in range(len(labels_data[i])):
                        labels.append(labels_data[i][j][:5])
                        preds.append(preds_data[i][j][:5])
        labels = np.array(labels)
        preds = np.array(preds)
        max_hamming_score = 0
        max_thres = 0
        for thresh in np.linspace(0, 1, 11):
                copy_preds = preds.copy()
                copy_preds[copy_preds >= thresh] = 1
                copy_preds[copy_preds < thresh] = 0
                if max_hamming_score < hamming_score(labels, copy_preds):
                        max_hamming_score = hamming_score(labels, copy_preds)
                        max_thres = thresh
                        max_hamming_loss = hamming_loss(labels, copy_preds)
        print(f"The metrics for {run_name} run are:")
        print("Thresh:", max_thres, "\tHamming Loss:", max_hamming_loss, "\tHamming Score:", max_hamming_score)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        
        mapping = {0: "Organisation", 1: "Location", 2: "Individual", 3: "Community", 4: "None"}
        for i in range(0, 5):
                print(f"{mapping[i]} Precision: {precision_score(labels[:, i], preds[:, i])} Recall: {recall_score(labels[:, i], preds[:, i])} F1 Score: {f1_score(labels[:, i], preds[:, i])}")
        print()
        file_count += 2