import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# Load and preprocess data
data = load_breast_cancer()
X = data.data
y = data.target

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)


def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Initialize variables to store the accumulated history
accumulated_history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

# Perform K-Fold Cross-Validation
fold_count = 0
for train_index, test_index in kf.split(X_scaled):
    # Splitting the data for training and testing
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Creating and training the model
    model = create_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping],
                        verbose=0)

    # Append the history from this fold to the accumulated history
    for key in history.history.keys():
        if fold_count == 0:  # Only initialize the lists once (for the first fold)
            accumulated_history[key] = [history.history[key]]
        else:
            accumulated_history[key].append(history.history[key])

    fold_count += 1

# Determine the maximum length of the history lists across all folds
max_length = max(max(len(hist) for hist in fold_history) for fold_history in accumulated_history.values())

# Pad the history lists so that they all have the same length
for key in accumulated_history.keys():
    for i in range(len(accumulated_history[key])):
        while len(accumulated_history[key][i]) < max_length:
            accumulated_history[key][i].append(accumulated_history[key][i][-1])

# Convert padded lists to numpy arrays for easier manipulation
for key in accumulated_history:
    accumulated_history[key] = np.array(accumulated_history[key])

# Calculate the average for each metric across all folds
average_history = {
    key: np.mean(accumulated_history[key], axis=0)
    for key in accumulated_history
}

# Plotting the averaged metrics
plt.figure(figsize=(12, 6))

# Average training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(average_history['accuracy'], label='Train Accuracy')
plt.plot(average_history['val_accuracy'], label='Validation Accuracy')
plt.title('Average Model Accuracy across Folds')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Average training & validation loss
plt.subplot(1, 2, 2)
plt.plot(average_history['loss'], label='Train Loss')
plt.plot(average_history['val_loss'], label='Validation Loss')
plt.title('Average Model Loss across Folds')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Sources:
# - Breast Cancer Wisconsin (Diagnostic) Data Set: UCI Machine Learning Repository - https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
# - Scikit-learn Machine Learning Library: https://scikit-learn.org/
# - Keras Neural Network Library: https://keras.io/
