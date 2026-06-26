from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

print("First Review Length :", len(X_train[0]))
print("Second Review Length:", len(X_train[1]))
print("Third Review Length :", len(X_train[2]))

review_lengths = [len(review) for review in X_train]

print("Maximum Review Length :", max(review_lengths))
print("Minimum Review Length :", min(review_lengths))

print("Average Review Length :", sum(review_lengths) / len(review_lengths))

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

print("\nAfter Padding")

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

print("\nFirst Review Length :", len(X_train[0]))
print("Second Review Length:", len(X_train[1]))