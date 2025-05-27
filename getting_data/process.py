import numpy as np
import pandas as pd

df = pd.read_csv('/Users/songsumin/Desktop/inquiry/getting_data/data.csv', header=None)

group_size = 10
n = len(df)
trimmed = df.iloc[:n - (n % group_size), :] 
grouped = trimmed.values.reshape(-1, group_size, df.shape[1]) 

averaged = grouped.mean(axis=1)  

averaged_df = pd.DataFrame(averaged)
averaged_df.to_csv('/Users/songsumin/Desktop/inquiry/getting_data/averaged_all_columns.csv', index=False, header=False)

print("완료: averaged_all_columns.csv 저장됨.")
