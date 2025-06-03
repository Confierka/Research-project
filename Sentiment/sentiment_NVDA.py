#В качестве источника моделей используется huggingface
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import json
from bs4 import BeautifulSoup
import pandas as pd

#Загрузка новостей

texts=[]
with open("doneNVDA.json","r",encoding='utf-8') as file:
    data = json.load(file)

len_file= len(data)
for item in data:
    if "data" in item and "attributes" in item["data"]:
        title = item["data"]["attributes"]["title"]
        html = item["data"]["attributes"]["content"]
        date = item["data"]["attributes"]["publishOn"]

        #Подгатавливаем soup
        soup = BeautifulSoup(html, 'html.parser')
        soup1 = BeautifulSoup(title, 'html.parser')

        # Извлекаем текст начала
        img_tag = soup.find('img')
        beggining = img_tag['alt'] if img_tag and 'alt' in img_tag.attrs else ""
        # Извлекаем загаловок и основной текст 
        for elem in soup(['figure', 'script', 'style', 'div']):
            elem.decompose()
        for elem in soup1(['figure', 'script', 'style', 'div']):
            elem.decompose()

        # Получаем чистый текст
        clean_title = soup1.get_text(separator=' ', strip=True)
        clean_text = soup.get_text(separator=' ', strip=True)

        clean_text = clean_title+' '+beggining+' '+clean_text
        #texts.append(clean_text)
        clean_date = date.split("T")[0]
        texts.append((clean_date,clean_text))


# Загрузка модели и токенизатора FinBert для анализа тональности
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
#Загрузка модели и токенизатора zero-shot классификатоор для анализа причастности новости
tokenizer1 = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model1 = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

# Создание пайплайнов
sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
classifier = pipeline("zero-shot-classification", model=model1, tokenizer=tokenizer1)

#Функция прчиастности новости к Акции
def is_about_stock(text, threshold=0.75):
    if len(text)>512:
        text = text[:512]
    candidate_labels = [        "Nvidia", 
        "GPU", 
        "graphics card", 
        "semiconductors", 
        "Jensen Huang", 
        "AI chips", 
        "Not related to Nvidia"]

    result = classifier(text, candidate_labels, multi_label=True)

    # Считаем связанным, если любая тесловская тема выше порога и выше "Not related"
    for label, score in zip(result['labels'], result['scores']):
        if label != "Not related to Nvidia" and score > threshold:
            return True
    return False

#Функция тональности новости
def get_sentiment(text):
    if len(text)>512:
        text = text[:512]
    return sentiment("This news may affect Nvidia's stock price. " + text)[0]['label']

#Функция общего анализа новости
def analyze_news(text):
    if len(text)>512:
        text = text[:512]
    if is_about_stock(text):
        sentiment = get_sentiment(text)
        return {"relevant": True, "sentiment": sentiment}
    else:
        return {"relevant": False, "sentiment": None}


sentiments = []

for i in range(1261):
    date, text = texts[i]
    data = analyze_news(text)
    relevance = data['relevant']
    sentiment_result  = data['sentiment']
    # print(text)
    # print(data)
    print(i)
    if relevance == True:
        sentiments.append((date,sentiment_result ))
        print("appened")
 
buff_df = pd.DataFrame(sentiments, columns=['date', 'sentiment'])
buff_df['date'] = pd.to_datetime(buff_df['date'])
buff_df.to_csv('NVIDIA_sentiment.csv', index=False)
