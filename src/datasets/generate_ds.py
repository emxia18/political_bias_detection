import numpy as np
from dataset import Dataset

allsides_dataset = Dataset('data/allsides_balanced_news_headlines-texts.csv')
data = allsides_dataset.get_data()

santana_map = {'left' : 'left', 'lean left' : 'left', 'center' : 'center', 'lean right': 'right', 'right' : 'right'}
santana_dataset = Dataset('data/mayobanexsantana/political-bias/versions/1/Political_Bias.csv', 'Title', 'Text', 'Bias', santana_map)
data.extend(santana_dataset.get_data())

np.save('src/datasets/full_dataset.npy', data)
print("Array saved successfully!")