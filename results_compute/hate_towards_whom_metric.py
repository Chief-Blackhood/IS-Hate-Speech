import os
import sys
import numpy as np
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score

NPY_FILE_PATH = "../npy_files/7Jun_vision"

filenames = os.listdir(path=NPY_FILE_PATH)
filenames = sorted(filenames, key=lambda x: x.split('_')[-1])
filenames = list(set([filename.split('_')[2] for filename in filenames]))

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


for filename in filenames:
        run_name = filename.split('.')[0]
        print(f"The metrics for {run_name} run are:")
        labels_data = np.load(f'{NPY_FILE_PATH}/test_labels_{filename}', allow_pickle=True)
        preds_data = np.load(f'{NPY_FILE_PATH}/test_preds_{filename}', allow_pickle=True)
        eval_labels_data = np.load(f'{NPY_FILE_PATH}/eval_labels_{filename}', allow_pickle=True)
        eval_preds_data = np.load(f'{NPY_FILE_PATH}/eval_preds_{filename}', allow_pickle=True)
        labels = []
        preds = []
        eval_labels = []
        eval_preds = []
        for i in range(len(labels_data)):
                for j in range(len(labels_data[i])):
                        labels.append(labels_data[i][j][:5])
                        preds.append(preds_data[i][j][:5])
        for i in range(len(eval_labels_data)):
                for j in range(len(eval_labels_data[i])):
                        eval_labels.append(eval_labels_data[i][j][:5])
                        eval_preds.append(eval_preds_data[i][j][:5])
        labels = np.array(labels)
        preds = np.array(preds)
        eval_labels = np.array(eval_labels)
        eval_preds = np.array(eval_preds)
        max_hamming_score = 0
        max_thres = 0
        for thresh in np.linspace(0, 1, 11):
                copy_preds = eval_preds.copy()
                copy_preds[copy_preds >= thresh] = 1
                copy_preds[copy_preds < thresh] = 0
                if max_hamming_score < hamming_score(eval_labels, copy_preds):
                        max_hamming_score = hamming_score(eval_labels, copy_preds)
                        max_thres = thresh
                        # max_hamming_loss = hamming_loss(eval_labels, copy_preds)
        copy_preds = preds.copy()
        copy_preds[copy_preds >= max_thres] = 1
        copy_preds[copy_preds < max_thres] = 0                
        
        print("Thresh:", max_thres, "\tHamming Loss:", hamming_loss(copy_preds, labels), "\tHamming Score:", hamming_score(copy_preds, labels))
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        
        mapping = {0: "Organisation", 1: "Location", 2: "Individual", 3: "Community", 4: "None"}
        for i in range(0, 5):
                print(f"{mapping[i]} Precision: {precision_score(labels[:, i], preds[:, i])} Recall: {recall_score(labels[:, i], preds[:, i])} F1 Score: {f1_score(labels[:, i], preds[:, i])}")
        print()