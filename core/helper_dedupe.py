"""
OPENS A CSV FILE AND DEDUPES BASED ON THE FIRST COLUMN
OVERWRITES THE CSV
"""
import pandas as pd

filePath = 'data/eric_standard.csv'

# Read CSV file into a pandas DataFrame
df = pd.read_csv(filePath)

# Drop duplicates based on the first column
df.drop_duplicates(subset=df.columns[0], keep='first', inplace=True)

# Save the deduplicated data back to the same CSV file
df.to_csv(filePath, index=False)
