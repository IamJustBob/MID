import pandas as pd

# Read the CSV file
df = pd.read_csv('sub-dataset.csv')

# Convert the 'label' column to string
df['label'] = df['label'].astype(str)

# Write the DataFrame back to CSV
df.to_csv('sub-dataset.csv', index=False)
