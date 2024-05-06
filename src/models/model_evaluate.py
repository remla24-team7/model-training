"""
Module for evaluating the neural network model trained.
"""

from keras.src.saving import load_model
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_curve, roc_auc_score)
from joblib import load
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

model = load_model('outputs/model.h5')

x_test = load('outputs/x_test.joblib')
y_test = load('outputs/y_test.joblib')

y_pred = model.predict(x_test, batch_size=1000)

# Convert predicted probabilities to binary labels
y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
y_test = y_test.reshape(-1, 1)

# Calculate classification report
report = classification_report(y_test, y_pred_binary, output_dict=True)
print('Classification Report:')
print(report)

df = pd.DataFrame(report).transpose()
df.to_csv('outputs/classification_report.csv')

# Calculate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred_binary)
print('Confusion Matrix:', confusion_mat)
print('Accuracy:', accuracy_score(y_test, y_pred_binary))

ax = sns.heatmap(confusion_mat, annot=True)

ax.figure.savefig('outputs/confusion_matrix.png')

# Calculate the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_binary)

# Calculate the area under the ROC curve
auc = roc_auc_score(y_test, y_pred_binary)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')  # Plot the diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(['ROC curve (AUC = {:.2f})'.format(auc)])
plt.savefig('outputs/roc_curve.png')
