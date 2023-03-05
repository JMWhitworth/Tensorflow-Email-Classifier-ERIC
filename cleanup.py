import pandas as pd
import string

# Replace `filename.csv` with the actual name of your CSV file
df = pd.read_csv('data/eric_standard.csv')

# Drop any rows that contain NaN
df = df.dropna(subset=[df.columns[1]])
df = df.dropna(subset=[df.columns[0]])

# Ensure all chars in first column are printable
df = df[df.iloc[:,0].str.contains('^[' + string.printable + ']*$')]

# Save the modified CSV file
df.to_csv('data/eric_standard_new.csv', index=False)
