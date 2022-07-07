from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import os
import numpy as np

NPY_FILE_PATH = "./npy_files"

filenames = os.listdir(path=NPY_FILE_PATH)
filenames = sorted(filenames, key=lambda x: x.split('_')[-1])

i=0
while i < len(filenames):
    run_name = filenames[i].split('.')[0].split('_')[-1]
    labels = np.load(f'{NPY_FILE_PATH}/{filenames[i]}')
    preds = np.load(f'{NPY_FILE_PATH}/{filenames[i+1]}')
    print(f"\nThe metrics for {run_name} run are:")
    print("     Precision:", precision_score(labels, preds, pos_label=1))
    print("     Recall:", recall_score(labels, preds, pos_label=1))
    print("     F1 score:", f1_score(labels, preds, pos_label=1))
    print("     Accuracy:", accuracy_score(labels, preds))
    i+=2