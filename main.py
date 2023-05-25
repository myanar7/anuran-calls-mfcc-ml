
# Write code to read a file named Frogs_MFCCs.csv in dataset folder
# Path: main.py
import pandas as pd
df = pd.read_csv('dataset/Frogs_MFCCs.csv')
print(df.head())

#Print the first 5 rows of the dataset

df.sample(5)