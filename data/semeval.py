import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(article_file, label_file, csv_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    for article in root.findall('article'):
        article_id = article.get('id')
        title = article.get('title')
        content = "".join(article.itertext()).strip()

        data.append([article_id, published_at, title, content])
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    dict = {}
    for article in root.findall('article'):
        article_id = article.get('id')
        article_label = article.get()

    df = pd.DataFrame(data, columns=['ID', 'Published Date', 'Title', 'Content'])

    df.to_csv(csv_file, index=False)
    print(f"XML successfully converted to CSV: {csv_file}")

xml_file = "data/articles-training-byarticle-20181122.xml" 
csv_file = "data/semeval.csv"
xml_to_csv(xml_file, csv_file)
