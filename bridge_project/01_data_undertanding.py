from tensorflow.keras.datasets import imdb

# Load IMDb dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Dataset size
print("Training Reviews :", len(X_train))
print("Training Labels  :", len(y_train))

print()

print("Testing Reviews  :", len(X_test))
print("Testing Labels   :", len(y_test))

# First review
print("\nFirst Review (Token IDs):\n")
print(X_train[0])

# First label
print("\nLabel:", y_train[0])


# Step3 decode review
# Get word dictionary
word_index = imdb.get_word_index()

# Reverse dictionary
reverse_word_index = {
    value + 3: key
    for key, value in word_index.items()
}

# Special tokens
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<OOV>"
reverse_word_index[3] = "<UNUSED>"

# Decode first review
decoded_review = " ".join(
    reverse_word_index.get(i, "?")
    for i in X_train[0]
)

print("\nDecoded Review:\n")
print(decoded_review)