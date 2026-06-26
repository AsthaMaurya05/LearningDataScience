from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

MAX_LENGTH = 200

X_train = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

from tensorflow.keras.layers import Input

model = Sequential([
    Input(shape=(200,)),
    Embedding(10000, 32),
    Bidirectional(LSTM(64)),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.add(
    Embedding(
        input_dim=10000,
        output_dim=32,
        input_length=200
    )
)
model.add(
    Bidirectional(
        LSTM(64)
    )
)
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)


y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))