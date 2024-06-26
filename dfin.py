import pandas as pd

df = pd.read_csv('data/41587_2020_793_MOESM3_ESM.csv')
df = df[df['partition']=='singles']

print(len(df),len(set(df['sequence'])))