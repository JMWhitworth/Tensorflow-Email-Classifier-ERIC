"""
@jackmwhit - 2022-02-21
EMAIL | RETRIEVER | IN | CODE
tf.__version__ = 2.8.0
"""

import tensorflow                               as tf
from tensorflow.keras.preprocessing.text        import Tokenizer
from tensorflow.keras.preprocessing.sequence    import pad_sequences
from tensorflow.keras                           import layers
import pandas                                   as pd
import string
import pickle

class Eric():
    def __init__(self, training=False, model_path="model", max_length=50, epochs=15):
        self.training = training
        self.model_path = model_path
        self.max_length = max_length
        self.epochs = epochs
        
        self.unique_characters = len(string.printable)
        
        if training:
            print("Creating a new model...")
            self.create_model()
        else:
            print("Loading existing model...")
            self.load_model()
    
    def create_model(self):
        print("Creating model...")
        self.load_training_data()
        self.build_model()
        self.train_model()
        self.save_model()
    
    def load_training_data(self):
        """
        Loads training data from two csv files
        expected in 'data/train.csv' and 'data/test.csv'
        """
        print("Loading training data...")
        trainingData = pd.read_csv("data/train.csv")
        testingData = pd.read_csv("data/test.csv")
        trainingData["label"] = trainingData.label.map(self.breakdown_words)
        testingData["label"]  = testingData.label.map(self.breakdown_words)
        self.train_sentences = trainingData.label.to_numpy()
        self.train_labels    = trainingData.target.to_numpy()
        self.test_sentences  = testingData.label.to_numpy()
        self.test_labels     = testingData.target.to_numpy()
        self.build_tokenizer()
        train_sequences = self.tokenizer.texts_to_sequences(self.train_sentences)
        test_sequences = self.tokenizer.texts_to_sequences(self.test_sentences)
        self.train_padded = pad_sequences(train_sequences, maxlen=self.max_length, padding="post", truncating="post")
        self.test_padded = pad_sequences(test_sequences, maxlen=self.max_length, padding="post", truncating="post")
    
    def build_tokenizer(self):
        """
        Tokenizer used as a key for encoding characters
        Will be saved in the model directory when completed
        """
        print("Building tokenizer...")
        self.tokenizer = Tokenizer(num_words=self.unique_characters)
        self.tokenizer.fit_on_texts(self.train_sentences)
    
    def build_model(self):
        print("Building model...")
        self.model = tf.keras.models.Sequential()
        self.model.add(layers.Embedding(self.unique_characters, 32, input_length=self.max_length))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.LSTM(64, dropout=0.1))
        self.model.add(layers.Dense(1, activation="sigmoid"))
        print(self.model.summary())
    
    def train_model(self):
        print("Training model...")
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        optim = tf.keras.optimizers.Adam(learning_rate=0.001)
        metrics = ["accuracy"]
        self.model.compile(loss=loss, optimizer=optim, metrics=metrics)
        self.model.fit(self.train_padded, self.train_labels, epochs=self.epochs, validation_data=(self.test_padded, self.test_labels), verbose=2)
    
    def save_model(self):
        print("Saving model...")
        self.model.save(self.model_path)
        with open(self.model_path+'/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_model(self):
        print("Loading model...")
        model = tf.keras.models.load_model(self.model_path)
        with open(self.model_path + '/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        self.model, self.tokenizer = model, tokenizer
    
    def model_summary(self):
        return(self.model.summary())
    
    def predict(self, input_list) -> list:
        """
        Takes input list and returns a prediction list
        """
        test = pd.DataFrame({"label": input_list})
        test["label"] = test.label.map(self.breakdown_words)
        test_sequences = self.tokenizer.texts_to_sequences(test.label.to_numpy())
        test_padded = pad_sequences(test_sequences, maxlen=self.max_length, padding="post", truncating="post")
        prediction = self.model.predict(test_padded)
        prediction = [1 if p > 0.5 else 0 for p in prediction]
        return(prediction)
    
    def breakdown_words(self, word) -> list:
        """
        Takes a string and returns it as a list of characters
        """
        output = ([char for char in str(word)])
        return(output)