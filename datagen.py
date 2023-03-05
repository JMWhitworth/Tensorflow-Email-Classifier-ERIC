import pandas as pd
import random

from core.Tokeniser import Tokeniser

filePath = 'data/eric_standard.csv'

df = pd.read_csv(filePath)
tk = Tokeniser()

i = 0

for i in range(0, 1000):
    nums = []
    for n in range(0, random.randrange(5,32)):
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
