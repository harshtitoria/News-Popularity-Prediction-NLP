import requests
from bs4 import BeautifulSoup
from newspaper import Article  
import csv 
import pandas as pd
import numpy as np

url = "https://www.newindianexpress.com"
page = requests.get(url)

soup = BeautifulSoup(page.text, 'html.parser')

articles = soup.findAll('a', class_="article_click")
news=[]
for row in articles:
    news.append(row['href'])
    #link = articles[row].find('a')['href']
    #news.append(link)
    
    
    
dataset=[]
for i in news:
    article = Article(i, language="en")
    article.download() 
    article.parse() 
    article.nlp() 
    data={}
    data['Title']=article.title
    data['Text']=article.text
    data['Summary']=article.summary
    data['Keywords']=article.keywords
    dataset.append(data)

#print(data)    
df=pd.DataFrame(dataset)


# Importing the dataset
uci_dataset = pd.read_csv('OnlineNewsPopularity.csv', quoting = 3, index_col = False)

#Cleaning the columns headers of whitespaces
arr = list(uci_dataset)
cleaned_columns = {x:x.lower().strip() for x in arr}
new_dataset = uci_dataset.rename(columns=cleaned_columns)

#We are removing features which are not most relevant for our model
x = new_dataset.drop(['url','shares', 'timedelta', 'lda_00','lda_01',
                  'lda_02','lda_03','lda_04','num_self_hrefs', 
                  'kw_min_min', 'kw_max_min', 'kw_avg_min',
                  'kw_min_max','kw_max_max','kw_avg_max',
                  'kw_min_avg','kw_max_avg','kw_avg_avg',
                  'self_reference_min_shares','self_reference_max_shares',
                  'self_reference_avg_sharess','rate_positive_words',
                  'rate_negative_words','abs_title_subjectivity',
                  'abs_title_sentiment_polarity'], axis = 1)
y = new_dataset['shares']

#Splitting the new_dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0) 

#Fitting the random forest regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#Comparison of y_test and y_pred
pred_result = pd.DataFrame(list(y_test), y_pred)
pred_result.reset_index(0, inplace=True)
pred_result.columns = ['Predicted share','Actual shares']

#Converting the crawled new according to UCI_Dataset Using NLP
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

stopwords=set(stopwords.words('english'))

#Tokenization:Tokenization is the process of tokenizing or splitting a string, text into a list of tokens
def tokenize(txt):
    return word_tokenize(txt)

#n_unique_tokens: Rate of unique words in the content
def n_unique_tokens(txt):
    txt=tokenize(txt)
    words = list(set(txt))   ##sets only store unique values
    n_unique_tokens=len(words)/len(txt)
    return n_unique_tokens

#average_token_length: Average length of the words in the content
def avg_token_length(txt):
    txt=tokenize(txt)
    length=[]
    for i in txt:
        length.append(len(i))
    return np.average(length)


#n_non_stop_words: Rate of non-stop words in the content
#n_non_stop_unique_tokens: Rate of unique non-stop words in content
def n_nonstop_words(txt):
    txt=tokenize(txt)
    nonstop_words = [i for i in txt if not i in stopwords]
    n_nonstop_words=len(nonstop_words)/len(txt)
    nonstop_unique_words = list(set(nonstop_words))
    n_nonstop_unique_tokens=len(nonstop_unique_words)/len(txt)
    return n_nonstop_words,n_nonstop_unique_tokens



import datefinder  #datefinder - extract dates from text
import datetime  

#from datetime import date 

#weekday #is_weekend
def day(txt):
    article_url=txt
    l1 = article_url.split("-")  #To remove the int value in url like "2140736"
    date_url = l1[0]             #which was getting was getting assigned as month
    if len(list(datefinder.find_dates(date_url)))>0:
        date_time=list(datefinder.find_dates(date_url))
        date=(str(date_time[0])).split()
        date=date[0]
        year, month, day = date.split('-')     
        weekday = datetime.date(int(year), int(month), int(day)) 
        return weekday.strftime("%A")  # ".strftime" gives a weekday from a date
    return "Monday"

#Polar words
positive_words=[]
negative_words=[]
def polarity(txt):
    tokenize_txt=tokenize(txt)
    for i in tokenize_txt:
        blob=TextBlob(i)
        polarity=blob.sentiment.polarity
        if polarity>0:
            positive_words.append(i)
        if polarity<0:
            negative_words.append(i)
    return positive_words,negative_words

#Polarity_rates
def rates(txt):
    txt=polarity(txt)
    positive_words=txt[0]
    negative_words=txt[1]
    global_rate_positive_words=(len(positive_words)/len(txt))/100
    global_rate_negative_words=(len(negative_words)/len(txt))/100
    positive_polarity=[]
    negative_polarity=[]
    for i in positive_words:
        blob_a=TextBlob(i)
        positive_polarity.append(blob_a.sentiment.polarity)
    for j in negative_words:
        blob_b=TextBlob(j)
        negative_polarity.append(blob_b.sentiment.polarity)
    min_positive_polarity=min(positive_polarity)
    min_negative_polarity=min(negative_polarity)
    max_positive_polarity=max(positive_polarity)
    max_negative_polarity=max(negative_polarity)
    avg_positive_polarity=np.average(positive_polarity)
    avg_negative_polarity=np.average(negative_polarity)
    return (global_rate_positive_words,global_rate_negative_words,
            avg_positive_polarity,min_positive_polarity,
            max_positive_polarity,avg_negative_polarity,
            min_negative_polarity,max_negative_polarity)

final_dataset=[]
for i in news:
    content={}
    article = Article(i, language="en")
    article.download() 
    article.parse()
    blob=TextBlob(article.text)
    #polarity=blob.sentiment.polarity
    title_blob=TextBlob(article.title)
    content['title']=article.title
    content['n_tokens_title']=len(tokenize(article.title))
    content['n_tokens_content']=len(tokenize(article.text))
    content['n_unique_tokens']=n_unique_tokens(article.text)
    content['n_non_stop_words']=n_nonstop_words(article.text)[0]
    content['n_non_stop_unique_tokens']=n_nonstop_words(article.text)[1]
    content['num_hrefs']=article.html.count("https://www.newindianexpress.com")
    content['num_imgs']=len(article.images)
    content['num_videos']=len(article.movies)
    content['average_token_length']=avg_token_length(article.text)
    content['num_keywords']=len(article.keywords)
    
    if "lifestyle" in article.url:
        content['data_channel_is_lifestyle']=1
    else:
        content['data_channel_is_lifestyle']=0
    if "entertainment" in article.url:
        content['data_channel_is_entertainment']=1
    else:
        content['data_channel_is_entertainment']=0
    if "business" in article.url:
        content['data_channel_is_bus']=1
    else:
        content['data_channel_is_bus']=0
    if "social media" or "facebook" or "whatsapp" in article.text.lower():
        data_channel_is_socmed=1
        data_channel_is_tech=0
        data_channel_is_world=0
    else:
        data_channel_is_socmed=0
    if ("technology" or "tech" in article.text.lower()) or ("technology" or "tech" in article.url):
        data_channel_is_tech=1
        data_channel_is_socmed=0
        data_channel_is_world=0
    else:
        data_channel_is_tech=0
    if "world" in article.url:
        data_channel_is_world=1
        data_channel_is_tech=0
        data_channel_is_socmed=0
    else:
        data_channel_is_world=0
        
    content['data_channel_is_socmed']=data_channel_is_socmed
    content['data_channel_is_tech']=data_channel_is_tech
    content['data_channel_is_world']=data_channel_is_world
    
    if day(i)=="Monday":
        content['weekday_is_monday']=1
    else:
        content['weekday_is_monday']=0
    if day(i)=="Tuesday":
        content['weekday_is_tuesday']=1
    else:
        content['weekday_is_tuesday']=0
    if day(i)=="Wednesday":
        content['weekday_is_wednesday']=1
    else:
        content['weekday_is_wednesday']=0
    if day(i)=="Thursday":
        content['weekday_is_thursday']=1
    else:
        content['weekday_is_thursday']=0
    if day(i)=="Friday":
        content['weekday_is_friday']=1
    else:
        content['weekday_is_friday']=0
    if day(i)=="Saturday":
        content['weekday_is_saturday']=1
        content['is_weekend']=1
    else:
        content['weekday_is_saturday']=0
    if day(i)=="Sunday":
        content['weekday_is_sunday']=1
        content['is_weekend']=1
    else:
        content['weekday_is_sunday']=0
        content['is_weekend']=0
        
    content['global_subjectivity']=blob.sentiment.subjectivity
    content['global_sentiment_polarity']=blob.sentiment.polarity
    content['global_rate_positive_words']=rates(article.text)[0]
    content['global_rate_negative_words']=rates(article.text)[1]
    content['avg_positive_polarity']=rates(article.text)[2]
    content['min_positive_polarity']=rates(article.text)[3]
    content['max_positive_polarity']=rates(article.text)[4]
    content['avg_negative_polarity']=rates(article.text)[5]
    content['min_negative_polarity']=rates(article.text)[6]
    content['max_negative_polarity']=rates(article.text)[7]    
    content['title_subjectivity']=title_blob.sentiment.subjectivity
    content['title_sentiment_polarity']=title_blob.sentiment.polarity
    final_dataset.append(content)

final_df=pd.DataFrame(final_dataset)
test_df=final_df.drop(['title'],axis=1)


predicted_shares = regressor.predict(test_df)

final_pred_result = pd.DataFrame(predicted_shares,final_df['title'])
final_pred_result.reset_index(0, inplace=True)
final_pred_result.columns = ['Title','Predicted shares']
print(final_pred_result)
#final_pred_result.to_csv("predicted_shares.csv")














