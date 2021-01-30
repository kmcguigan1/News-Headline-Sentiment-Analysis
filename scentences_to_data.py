from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

SENTENCES = [
    'I love my dog',
    'I love my cat',
    'You love my dog',
    'Do you like my dog?'
]

TEST_SENTENCES = [
    'I really love my dog',
    'I love wolves'
]

def make_tokenizer(sentences):
    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    return tokenizer

def make_sequences(tokenizer, sentences):
    sequences = tokenizer.texts_to_sequences(sentences)
    return sequences

def pad_sequence(sequences):
    sequences = pad_sequences(sequences, padding='post', truncating='post', maxlen=10)
    return sequences

tokenizer = make_tokenizer(SENTENCES)
print(tokenizer.word_index)

print("\nSequences")
sequences = make_sequences(tokenizer, SENTENCES)
print(sequences)
test_sequences = make_sequences(tokenizer, TEST_SENTENCES)
print(test_sequences)

print("\nPadded sequences")
sequences = pad_sequence(sequences)
print(sequences)
test_sequences = pad_sequence(test_sequences)
print(test_sequences)