"""
LOADS CSV AND ADDS DATA TO THE BOTTOM OF IT WHICH WON'T BE EMAILS
USED TO PADD TRAINING DATA
"""
import pandas as pd
import random

#Run from main directory to use this import
from Tokeniser import Tokeniser

#File path of data
filePath = 'data/eric_standard.csv'

#Load data & create tokeniser for encoding/decoding
df = pd.read_csv(filePath)
tk = Tokeniser()

#Amount of entries to generate
rowsToCreate = 10000
#Minimum number of characters per entry
minEntryLength = 5
#Maximum number of characters per entry
maxEntryLength = 32

for i in range(0, rowsToCreate):
    nums = []
    for n in range(0, random.randrange(minEntryLength,maxEntryLength)):
        nums.append(random.randrange(1, len(tk.characters)))
    text = tk.decode(nums)
    if i % 100 == 0:
        print(f"{i}: {text} = {nums}")
    
    # Create a new row with 'nums' and 0 as the values
    new_row = [text, 0]
    
    # Append the new row to the dataframe
    df.loc[len(df)] = new_row
    
# Save the dataframe to a CSV file
df.to_csv(filePath, index=False)
