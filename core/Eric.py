
"""
@jackwritescode - 2023-03-04
EMAIL | RETRIEVER | IN | CODE
tf.__version__ = 2.8.0
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
        data = pd.read_csv(dataPath).iloc[:10000]
        
        #Encode the labels and save to numpy array
        data['label'] = data.label.map(self.tokeniser.encode)
        data['label'] = data.label.map(self.tokeniser.normalise)
        data = data.to_numpy()
        
        #Enforce datatypes of floats for labels and ints for targets
        dataLabels = [[float(x) for x in seq] for seq in data[:, 0]]
        dataTargets = data[:, 1]
        dataTargets = np.where(dataTargets == 'NaN', 0, dataTargets.astype('int')).tolist()
        
        #Divide into training and testing datasets
        split = int(len(data)*0.1)
        trainingLabels = dataLabels[split:]
        trainingTargets = dataTargets[split:]
        testingLabels = dataLabels[:split]
        testingTargets = dataTargets[:split]
        

        print(len(trainingLabels))
        print(len(trainingTargets))
        input("Pause")

        #Build model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(len(self.tokeniser.characters), 32, input_length=self.tokeniser.paddingSize),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, dropout=0.1),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        model.fit(trainingLabels, trainingTargets, epochs=40, validation_data=(testingLabels, testingTargets), verbose=2)
        model.evaluate([self.tokeniser.encode('jack@gmail.com')])
    
    def loadModel(self, dataPath:str) -> None:
        """
        USES THE PATH TO LOAD A PRE-TRAINED MODEL
        """
        print("Will load model")
    
    def predict(self, potentialEmail:str) -> bool:
        """
        TEST THE POTENTIAL EMAIL AGAINST THE MODEL
        RETURNS BOOL OF PREDICTION
        """
        return 1
