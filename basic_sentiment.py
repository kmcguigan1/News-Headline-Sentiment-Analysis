import requests

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Activation
from tensorflow.keras.activations import relu, sigmoid

from tensorflow.keras.callbacks import EarlyStopping

DATASET_FILE = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json"
VOCAB_SIZE = 10_000
EMBEDDING_DIM = 16
MAX_LEN = 150
NUM_EPOCHS = 100
BATCH_SIZE = 32

MY_TEST_SCENTENCES = [
    "The weather today is sunny",
    "granny starting to fear spiders in the garden might be real"
]

def read_json(dataset_url):
    # send the get request to the server
    response = requests.get(dataset_url)
    # convert the response to json format
    datastore = response.json()
    return datastore

# have this make a tf dataset later on
def read_dataset(dataset_url=DATASET_FILE):
    # get the data
    datastore = read_json(dataset_url)
    # initialize the data containers
    sentences, labels, urls = [], [], []
    # iterate over the datastore and add items to respective lists
    for item in datastore:
        # add the news headline
        sentences.append(item['headline'])
        # add the target label
        labels.append(item['is_sarcastic'])
        # add the url to the article
        urls.append(item['article_link'])
    return sentences, labels, urls
    
def split_data(sentences, labels):
    # get the total dataset size
    datasize = len(labels)
    # get the partitions to split on
    train_split = int(datasize*0.7)
    val_split = int(datasize*0.15) + train_split
    # now split the data
    train_sentences, train_labels = sentences[:train_split], labels[:train_split]
    val_sentences, val_labels = sentences[train_split:val_split], labels[train_split:val_split]
    test_sentences, test_labels = sentences[val_split:], labels[val_split:]
    # return the split data
    return train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels

def make_tokenizer(sentences, vocab_size):
    # create the tokenizer on the passed in data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    return tokenizer

def make_sequences(tokenizer, sentences):
    sequences = tokenizer.texts_to_sequences(sentences)
    return sequences

def pad_sequence(sequences, max_length):
    sequences = pad_sequences(sequences, padding='post', truncating='post', maxlen=max_length)
    return sequences

def sentences_to_sequences(tokenizer, sentences, max_length):
    sequences = make_sequences(tokenizer, sentences)
    sequences = pad_sequences(sequences, max_length)
    return sequences

def sequences_to_dataset(sequences, labels, batch_size):
    # save the overall datasize
    dataset_size = len(labels)
    # create the dataset
    dataset = tf.data.Dataset.from_tensor_slices((sequences, labels))
    # shuffle and optimize the dataset
    dataset = dataset.cache().shuffle(dataset_size + 1).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def create_model(max_length, vocab_size, embedding_dim):
    # setup the model structure
    text_input = Input(shape=(max_length,), name="text")
    x = Embedding(vocab_size, embedding_dim, input_length=max_length)(text_input)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32)(x)
    x = Activation(relu)(x)
    x = Dense(1)(x)
    target_output = Activation(sigmoid, name="target")(x)
    # create the actual model
    model = tf.keras.Model(inputs=text_input, outputs=target_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, train_ds, val_ds, num_epochs):
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1, mode='min', restore_best_weights=True)
    hist = model.fit(train_ds, epochs=num_epochs, validation_data=val_ds, verbose=2, callbacks=[early_stopping])
    print("\n")
    return hist

def evaluate_model(model, test_ds):
    ev = model.evaluate(test_ds)
    print("\n")
    print("Evaluation")
    print(f"Loss: {ev[0]}")
    print(f"Accuracy: {ev[1]}")
    print("\n")
    return ev

def graph_results(hist):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 12))
    epochs = hist.epoch
    loss = hist.history['loss']
    acc = hist.history['accuracy']
    val_loss = hist.history['val_loss']
    val_acc = hist.history['val_accuracy']
    axs[0].plot(epochs, loss, label='loss')
    axs[0].plot(epochs, val_loss, label='val_loss')
    axs[1].plot(epochs, acc, label='acc')
    axs[1].plot(epochs, val_acc, label='val_acc')
    axs[0].grid(True, which='both', axis='both')
    axs[1].grid(True, which='both', axis='both')
    axs[0].set_ylabel('Loss')
    axs[1].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[1].set_xlabel('Epoch')
    axs[0].legend()
    axs[1].legend()
    fig.savefig("C:\\Users\\kiern\\MyFolders\\Code\\GitRepositories\\News-Headline-Sentiment-Analysis\\results.png")
    plt.show()
    return

def predict(model, tokenizer, predict_sentences, max_length):
    categories = {0:"Not Sarcastic", 1:"Is Sarcastic"}
    sequences = sentences_to_sequences(tokenizer, predict_sentences, max_length)
    predictions = model.predict(sequences)
    for sentence, prediction in zip(predict_sentences, predictions):
        category_prediction = categories[int(round(prediction[0], 0))]
        print(f"Prediction: {category_prediction}, confidence: {prediction}, sentence: {sentence}")
    return

def main():
    # get the data
    sentences, labels, urls = read_dataset()
    # break the data up into a train, val, and test split
    train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels = split_data(sentences, labels)
    # get a tokenizer trained on the training dataset
    tokenizer = make_tokenizer(train_sentences, VOCAB_SIZE)
    # get the padded and tokenized sequences
    train_sequences = sentences_to_sequences(tokenizer, train_sentences, MAX_LEN)
    val_sequences = sentences_to_sequences(tokenizer, val_sentences, MAX_LEN)
    test_sequences = sentences_to_sequences(tokenizer, test_sentences, MAX_LEN)
    # get datasets
    train_dataset = sequences_to_dataset(train_sequences, train_labels, BATCH_SIZE)
    val_dataset = sequences_to_dataset(val_sequences, val_labels, BATCH_SIZE)
    test_dataset = sequences_to_dataset(test_sequences, test_labels, BATCH_SIZE)
    # create the model
    model = create_model(MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM)
    # train the model
    hist = train_model(model, train_dataset, val_dataset, NUM_EPOCHS)
    # evaluate the model
    ev = evaluate_model(model, test_dataset)
    # graph the results
    graph_results(hist)
    # predict on the model
    #predict(model, tokenizer, MY_TEST_SCENTENCES, MAX_LEN)
    return

main()

























