import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer

preprocessor = DataPreprocessor()

allsides_data = preprocessor.load_data('data/allsides_balanced_news_headlines-texts.csv')
santana_map = {'left': 'left', 'lean left': 'left', 'center': 'center', 'lean right': 'right', 'right': 'right'}
santana_data = preprocessor.load_data('data/mayobanexsantana/political-bias/versions/1/Political_Bias.csv', 
                                      title_name="Title", text_name="Text", bias_name="Bias", bias_map=santana_map)

full_data = allsides_data + santana_data

train_data, val_data = train_test_split(full_data, test_size=0.2, random_state=42)

train_data = preprocessor.load_data(train_data)
val_data = preprocessor.load_data(val_data)
