import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

SENTENCES = [
    'I love dogs',
    'I love cats'
]

def make_tokenizer(sentences):
    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(sentences)
    return tokenizer

tokenizer = make_tokenizer(SENTENCES)
print("word index: ", tokenizer.word_index)
print("index word: ", tokenizer.index_word)
print("word counts: ", tokenizer.word_counts)
print("word_docs: ", tokenizer.word_docs)
print("filters: ", tokenizer.filters)
print("split: ", tokenizer.split)
print("lower: ", tokenizer.lower)
print("num words: ", tokenizer.num_words)
print("document count: ", tokenizer.document_count)
print("char level: ", tokenizer.char_level)
print("oov token: ", tokenizer.oov_token)
print("index docs: ", tokenizer.index_docs)