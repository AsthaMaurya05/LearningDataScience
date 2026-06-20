from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training Images:", X_train.shape)
print("Training Labels:", y_train.shape)

print("Testing Images:", X_test.shape)
print("Testing Labels:", y_test.shape)

print(X_train[0])
print()

print("Label:", y_train[0])


import matplotlib.pyplot as plt

plt.imshow(X_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.show()

# preprocessing step

# Normalize Pixels

X_train = X_train / 255.0
X_test = X_test / 255.0

print(X_train.min())
print(X_train.max())


# Flatten Images
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

print(X_train.shape)
print(X_test.shape)


# One-Hot Encoding
from tensorflow.keras.utils import to_categorical

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

print(y_train[0])
print(y_train_encoded[0])

print(y_train_encoded.shape)
print(y_test_encoded.shape)


# Build ANN Architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
from tensorflow.keras import Input

model = Sequential([
    Input(shape=(784,)),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])
model.summary()


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train_encoded,
    epochs=5,
    batch_size=100,
    validation_split=0.2
)

# Test Set Evaluation

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test_encoded
)

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# Analyzing Model
# step 1: Generate Predictions
y_pred_prob = model.predict(X_test)

print(y_pred_prob.shape)
print(y_pred_prob[0])

# Step 2: Convert Probabilities → Final Predictions
import numpy as np

y_pred = np.argmax(y_pred_prob, axis=1)

print(y_pred[:10])
print(y_test[:10])

# Step 3: Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("MNIST Confusion Matrix")

plt.show()

misclassified = np.where(y_pred != y_test)[0]

print("Total Wrong Predictions:", len(misclassified))
plt.figure(figsize=(12,8))

for i in range(9):

    idx = misclassified[i]

    plt.subplot(3,3,i+1)

    plt.imshow(
        X_test[idx].reshape(28,28),
        cmap="gray"
    )

    plt.title(
        f"Actual:{y_test[idx]} Pred:{y_pred[idx]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()