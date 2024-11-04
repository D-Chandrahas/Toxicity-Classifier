import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data/data.csv')

train_data, test_data = train_test_split(data, test_size=0.1, shuffle=True)
valid_data, test_data = train_test_split(test_data, test_size=0.5, shuffle=True)

train_data.sort_values("text", key = lambda s: s.str.len(), inplace=True)
valid_data.sort_values("text", key = lambda s: s.str.len(), inplace=True)
test_data.sort_values("text", key = lambda s: s.str.len(), inplace=True)

# train_data.to_csv('./data/train_data.csv', index=False)
# valid_data.to_csv('./data/valid_data.csv', index=False)
# test_data.to_csv('./data/test_data.csv', index=False)
