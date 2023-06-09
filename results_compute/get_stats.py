from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import os
import numpy as np

NPY_FILE_PATH = "../npy_files/7Jun_vision"

filenames = os.listdir(path=NPY_FILE_PATH)
filenames = sorted(filenames, key=lambda x: x.split('_')[-1])
filenames = list(set([filename.split('_')[2] for filename in filenames]))

for filename in filenames:
    run_name = filename.split('.')[0]
    labels_data = np.load(f'{NPY_FILE_PATH}/test_labels_{filename}', allow_pickle=True)
    preds_data = np.load(f'{NPY_FILE_PATH}/test_preds_{filename}', allow_pickle=True)
    labels = []
    preds = []
    for i in range(len(labels_data)):
        for j in range(len(labels_data[i])):
            labels.append(labels_data[i][j][-1])
            preds.append(preds_data[i][j][-1])
    labels = np.array(labels)
    preds = np.array(preds)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    
    # preds = 1/(1 + np.exp(-preds))
    print(f"\nThe metrics for {run_name} run are:")
    print("     Precision:", precision_score(labels, preds, pos_label=1))
    print("     Precision for (Non Hate):", precision_score(labels, preds, pos_label=0))
    print("     Recall:", recall_score(labels, preds, pos_label=1))
    print("     Recall for (Non Hate):", recall_score(labels, preds, pos_label=0))
    print("     F1 score:", f1_score(labels, preds, pos_label=1))
    print("     F1 score for (Non Hate):", f1_score(labels, preds, pos_label=0))
    print("     Accuracy:", accuracy_score(labels, preds))