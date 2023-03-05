
"""
@jack-writes-code - 2023-03-05
EMAIL | RETRIEVER | IN | CODE
Python version = 3.11.2
tf.__version__ = 2.12.0-rc0
"""

import tensorflow       as tf
from tensorflow.keras   import layers

import pandas           as pd
import numpy            as np

from core.Tokeniser     import Tokeniser


class Eric:
    def __init__(self, modelPath:str='', training:bool=False, dataPath:str='data.csv') -> None:
        self.modelPath = modelPath
        self.training = training
        self.dataPath = dataPath
        self.tokeniser = Tokeniser(paddingSize=32)
        
        if training:
            self.trainModel(dataPath)
        else:
            self.loadModel(modelPath)
    
    def trainModel(self, dataPath:str) -> None:
        #Load in the data
        data = pd.read_csv(dataPath)#.iloc[:100000]
        
        #Shuffle
        data = np.random.shuffle(data.values)
        
        #Encode the labels and save to numpy array
        data['label'] = data.label.map(self.tokeniser.encode)
        data = data.to_numpy()
        
        #Enforce datatypes of floats for labels and ints for targets
        dataLabels = [[float(x) for x in seq] for seq in data[:, 0]]
        dataTargets = data[:, 1].astype(int).tolist()
        
        #Divide into training and testing datasets
        split = int(len(data)*0.15)
        trainingLabels = dataLabels[split:]
        trainingTargets = dataTargets[split:]
        testingLabels = dataLabels[:split]
        testingTargets = dataTargets[:split]
        
        #Build model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(len(self.tokeniser.characters), 32, input_length=self.tokeniser.paddingSize),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.LSTM(64, dropout=0.1),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        print(self.model.summary())
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        self.model.fit(
            trainingLabels,
            trainingTargets,
            epochs=4,
            validation_data=(testingLabels, testingTargets),
            verbose=2
        )
        
        while True:
            print(self.predict(self.tokeniser.encode(input("Enter something to test: "))))
        
    def loadmodel(self, dataPath:str) -> None:
        """
        USES THE PATH TO LOAD A PRE-TRAINED MODEL
        """
        print("Will load model")
    
    def predict(self, potentialEmail:str) -> bool:
        
        # It can't be an email without the @ sign
        if '@' not in potentialEmail:
            return False
        
        # Run input against the model
        prediction = self.model.predict([self.tokeniser.encode(potentialEmail)])
        prediction = [1 if p > 0.5 else 0 for p in prediction][0]
        
        # If model says no but it ends with one of these, assume yes.
        emailEnders = ['.com', '.co.uk', '.org', '.net', '.us', '.co']
        for item in emailEnders:
            if potentialEmail.endswith(item) and not prediction:
                prediction = True
        
        return(prediction)
