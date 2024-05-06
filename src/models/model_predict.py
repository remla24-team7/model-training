from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, ConfusionMatrixDisplay
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_preprocessing import train_test_split
import os


x_train, x_val, x_test, y_train, y_val, y_test, char_index = train_test_split()

# Load the model
model = tf.keras.models.load_model('model.h5')

print('Model loading')
# Predict on test data
y_pred = model.predict(x_test)

# Convert predicted probabilities to binary labels
y_pred_binary = (np.array(y_pred) > 0.5).astype(int)

# Calculate classification report
report = classification_report(y_test, y_pred_binary)
print('Classification Report:')
print(report)


# Calculate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred_binary)
print('Confusion Matrix:', confusion_mat)

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=['phishing', 'legitimate'])
disp.plot()
plt.show


'''
# Plot confusion matrix
plt.imshow(confusion_mat, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# plt.xticks(np.arange(len(char_index)), char_index, rotation=90)
# plt.yticks(np.arange(len(char_index)), char_index)
plt.tight_layout()
'''
# Save the figure
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output'))
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
#plt.show()

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print('Accuracy:', accuracy)