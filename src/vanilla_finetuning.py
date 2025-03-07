from bias_dataset import DataPreprocessor, BiasDataset
from bias_trainer import BiasTrainer
from sklearn.model_selection import train_test_split

preprocessor = DataPreprocessor()

allsides_data = preprocessor.load_data('data/allsides_balanced_news_headlines-texts.csv')
santana_map = {'left': 'left', 'lean left': 'left', 'center': 'center', 'lean right': 'right', 'right': 'right'}
santana_data = preprocessor.load_data('data/mayobanexsantana/political-bias/versions/1/Political_Bias.csv', 
                                      title_name="Title", text_name="Text", bias_name="Bias", bias_map=santana_map)

full_data = allsides_data + santana_data

train_data, val_data = train_test_split(full_data, test_size=0.2, random_state=42)

train_encoded, label_mapping = preprocessor.encode_data(train_data)
val_encoded, _ = preprocessor.encode_data(val_data)

train_dataset = BiasDataset(train_encoded)
val_dataset = BiasDataset(val_encoded)

trainer = BiasTrainer(train_dataset, val_dataset)
trainer.train()

trainer.evaluate()
