
"""
@jackwritescode - 2023-03-03
EMAIL | RETRIEVER | IN | CODE
tf.__version__ = 2.8.0
"""

import tensorflow                               as tf
from tensorflow.keras.preprocessing.text        import Tokenizer
from tensorflow.keras.preprocessing.sequence    import pad_sequences
from tensorflow.keras                           import layers
import pandas                                   as pd
import string


#Get data
data = pd.read_csv('data.csv').iloc[:1000]

#Fix 0's being null in target column
data['target'] = data['target'].fillna(0).astype(int)

# Calculate the number of rows in the data
num_rows = len(data)

# Calculate the number of rows to include in the training set
num_train = int(0.9 * num_rows)

# Randomly select the rows to include in the training set
train_indices = pd.Series(range(num_rows)).sample(n=num_train, random_state=42)

# Split the data into training and validation sets
dataTrain = data.iloc[train_indices].label
dataTest = data.drop(train_indices).label
dataTrainTargs = data.iloc[train_indices].target
dataTestTargs = data.drop(train_indices).target

# Set maximum length of input
maxLength = 0
for index, row in data.iterrows():
    if len(row['label']) > maxLength:
        maxLength = len(row['label'])

# Create key-value map for characters to integers
tokeniser = {}
unique_characters = string.printable
for i, item in enumerate(unique_characters):
    tokeniser[item] = i

# Returns the given input as a tokenised version
def tokenise(input, tokeniser=tokeniser, maxLength=maxLength):
    output = []
    for character in input:
        output.append(tokeniser[character])
    while len(output) < maxLength:
        output.append(0)
    return output

word = tokenise('hello!')
print(word)
print(len(word))



input("stop")

model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(len(unique_characters), 32, input_length=maxLength),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.layers.LSTM(64, dropout=0.1),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.fit(dataTrain, dataTrainTargs, epochs=10, validation_data=(dataTest, dataTestTargs), verbose=2)
model.evaluate([tokenise('jack@gmail.com')])
