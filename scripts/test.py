import pandas as pd
train = pd.read_csv('data/splits/train_ratings.csv')
top_users = train.groupby('userId').size().sort_values(ascending=False).head(10)
print(top_users)