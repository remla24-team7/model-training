# evaluate_model.py
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from joblib import load
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_model(model_path, x_test_path, y_test_path, test):
    model = load_model(model_path)
    x_test = load(x_test_path)
    y_test = load(y_test_path)
    y_pred = model.predict(x_test, batch_size=1000)
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test.reshape(-1, 1)

    # If test, only use first 10 datapoints
    if test:
        x_test = x_test[:10]
        y_test = y_test[:10]
        y_pred = model.predict(x_test, batch_size=1000)
        y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
        y_test = y_test.reshape(-1, 1)

    report = classification_report(y_test, y_pred_binary, output_dict=True)
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    accuracy = accuracy_score(y_test, y_pred_binary)
    auc = roc_auc_score(y_test, y_pred_binary)

    ax = sns.heatmap(confusion_mat, annot=True)
    plt.savefig('outputs/confusion_matrix.png')

    plt.figure()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_binary)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(['ROC curve (AUC = {:.2f})'.format(auc)])
    plt.savefig('outputs/roc_curve.png')

    return report, confusion_mat, accuracy, auc

if __name__ == "__main__":
    model_path = 'outputs/model.h5'
    x_test_path = 'outputs/x_test.joblib'
    y_test_path = 'outputs/y_test.joblib'
    report, confusion_mat, accuracy, auc = evaluate_model(model_path, x_test_path, y_test_path,test=False)
    print('Classification Report:', report)
    print('Confusion Matrix:', confusion_mat)
    print('Accuracy:', accuracy)
    print('AUC:', auc)
