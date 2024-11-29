import re
import string
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt

# Dataset
dataset = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms.",
]

# Custom standardization: Lowercase and remove punctuation
def custom_standardization_fn(string_tensor):
    lowercase_string = tf.strings.lower(string_tensor)
    return tf.strings.regex_replace(lowercase_string, f"[{re.escape(string.punctuation)}]", "")

# Custom split: Tokenize into words
def custom_split_fn(string_tensor):
    return tf.strings.split(string_tensor)

# Create TextVectorization layer
text_vectorization = layers.TextVectorization(
    output_mode="int",
    standardize=custom_standardization_fn,
    split=custom_split_fn,
)

# Adapt the layer to the dataset
text_vectorization.adapt(dataset)

# Get vocabulary and display it
vocabulary = text_vectorization.get_vocabulary()
print("Vocabulary:", vocabulary)

# Test sentence
test_sentence = "I write, rewrite, and still rewrite again"
encoded_sentence = text_vectorization(test_sentence)
print("Encoded Sentence:", encoded_sentence.numpy())

# Decode sentence using the vocabulary
inverse_vocab = {i: word for i, word in enumerate(vocabulary)}
decoded_sentence = " ".join(inverse_vocab[int(i)] for i in encoded_sentence)
print("Decoded Sentence:", decoded_sentence)

# Calculate word frequencies
word_counts = [0] * len(vocabulary)
for line in dataset:
    encoded_line = text_vectorization(line)
    for word_idx in encoded_line:
        word_counts[word_idx] += 1

# Plot word frequency
plt.figure(figsize=(10, 6))
plt.bar(vocabulary, word_counts, color='skyblue')
plt.title("Word Frequency in Vocabulary")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("word_frequency.png")
plt.show()

# Compute sequence lengths
sequence_lengths = [len(text_vectorization(line)) for line in dataset]

# Plot sequence length distribution
plt.figure(figsize=(8, 5))
plt.hist(sequence_lengths, bins=len(set(sequence_lengths)), color='orange', edgecolor='black')
plt.title("Distribution of Sequence Lengths")
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("sequence_length_distribution.png")
plt.show()
