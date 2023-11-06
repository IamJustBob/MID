import pandas as pd

# Load the dataset
df = pd.read_csv('dataset.csv')

# Get the unique classes
classes = df['label'].unique()

# Print the number of classes
print(f'There are {len(classes)} classes.')

# Print each class
for i, class_name in enumerate(classes, 1):
    print(f'Class {i}: {class_name}')
