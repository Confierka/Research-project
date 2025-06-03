#В качестве источника моделей используется huggingface
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import json
from bs4 import BeautifulSoup
import pandas as pd

#Загрузка новостей

texts=[]
with open("doneTSLA.json","r",encoding='utf-8') as file:
    data = json.load(file)

len_Tesla= len(data)
for i in range(len_Tesla):
    title = data[i]["data"]["attributes"]["title"]
    html = data[i]["data"]["attributes"]["content"]
    date = data[i]["data"]["attributes"]["publishOn"]

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

#Функция прчиастности новости к Тесле
def is_about_tesla(text, threshold=0.8):
    if len(text)>512:
        text = text[:512]
    candidate_labels = [ "Tesla","Electric vehicles","Autonomous driving","Elon Musk","EV market","Clean energy","Not related to Tesla"]

    result = classifier(text, candidate_labels, multi_label=True)

    # Считаем связанным, если любая тесловская тема выше порога и выше "Not related"
    for label, score in zip(result['labels'], result['scores']):
        if label != "Not related to Tesla" and score > threshold:
            return True
    return False

#Функция тональности новости
def get_sentiment(text):
    if len(text)>512:
        text = text[:512]
    return sentiment("This news may affect Tesla's stock price. " + text)[0]['label']

#Функция общего анализа новости
def analyze_news(text):
    if len(text)>512:
        text = text[:512]
    if is_about_tesla(text):
        sentiment = get_sentiment(text)
        return {"relevant": True, "sentiment": sentiment}
    else:
        return {"relevant": False, "sentiment": None}


TSLA_sentiment = []
for i in range(len_Tesla):
    date, text = texts[i]
    #print(text)
    #print(analyze_news(text[i])[""])
    data = analyze_news(text)
    relevance = data['relevant']
    sentiment_result  = data['sentiment']
    print(i)
    if relevance == True:
        TSLA_sentiment.append((date,sentiment_result ))
        print("appened")
        
# for x in TSLA_sentiment:
#     print(x)
df_TSLA_sentiment = pd.DataFrame(TSLA_sentiment, columns=['date', 'sentiment'])
df_TSLA_sentiment['date'] = pd.to_datetime(df_TSLA_sentiment['date'])
df_TSLA_sentiment.to_csv('sentiment.csv', index=False)

#Ошибки модели
#16 новость
# Election jolts: Chinese EV fall while Tesla soars...
# ➤ negative, 0.97
# ❗ Ошибка модели — Tesla растёт, а не падает. Метка должна быть positive, ошибка!
#23 Новость
# Tesla crosses $1T market cap...
# ➤ negative, 0.94
# ❗ Ошибка модели — рост капитализации Tesla — это позитивная новость. Ошибка!

#Неточности модели
# Пограничные / Неоднозначные случаи:
#29 новось
# "Caterpillar, Tesla are top picks for 'red wave'" – neutral ❓
# Может трактоваться как позитив — рекомендации аналитиков. Но модель метит нейтрально. Возможна ошибка.


#KNN CNN RNN - Baseline
#LSTM GRU 