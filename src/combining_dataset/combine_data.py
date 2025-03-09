from bias_dataset import DataPreprocessor
import numpy as np
import pandas as pd

preprocessor = DataPreprocessor()

allsides_data = preprocessor.load_data('data/allsides_balanced_news_headlines-texts.csv')
santana_map = {'left': 'left', 'lean left': 'left', 'center': 'center', 'lean right': 'right', 'right': 'right'}
santana_data = preprocessor.load_data('data/mayobanexsantana/political-bias/versions/1/Political_Bias.csv', 
                                      title_name="Title", text_name="Text", bias_name="Bias", bias_map=santana_map)
pol_bias_data = preprocessor.load_data("data/political_bias_data_title.csv", 
                                      title_name="Title", text_name="Text", bias_name="Label")
print(len(allsides_data), len(santana_data), len(pol_bias_data))
full_data = allsides_data + santana_data + pol_bias_data
print(len(full_data))
df = pd.DataFrame(full_data, columns = ['title', 'text', 'label'])

df = df[df['text'].apply(len) >= 100]

df = df.reset_index(drop=True)

min_count = df['label'].value_counts().min()

df = df.groupby('label').apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)

print(df['label'].value_counts())
print(len(df))

pd.set_option('display.max_colwidth', None)

# print(df.iloc[0]['text'])
df.to_csv('src/combined_data.csv', index=False)
