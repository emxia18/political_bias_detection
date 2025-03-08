from bias_trainer import BiasEvaluator
from bias_dataset import DataPreprocessor
from sklearn.model_selection import train_test_split

MODEL_NAME = "emxia18/bias-vanilla"

evaluator = BiasEvaluator(MODEL_NAME)

preprocessor = DataPreprocessor()

allsides_data = preprocessor.load_data('data/allsides_balanced_news_headlines-texts.csv')
santana_map = {'left': 'left', 'lean left': 'left', 'center': 'center', 'lean right': 'right', 'right': 'right'}
santana_data = preprocessor.load_data('data/mayobanexsantana/political-bias/versions/1/Political_Bias.csv', 
                                      title_name="Title", text_name="Text", bias_name="Bias", bias_map=santana_map)

full_data = allsides_data + santana_data

_, val_data = train_test_split(full_data, test_size=0.1, random_state=42)
texts = [f"{title} {text}" for title, text, _ in val_data]
print(len(texts))
# print(texts[0])
word_importance_df = evaluator.analyze_word_importance(texts)

