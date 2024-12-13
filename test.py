import pandas as pd
import pickle

with open('dataset.pkl', 'rb') as file:
    df_filtered1 = pickle.load(file)

print(df_filtered1.head())
