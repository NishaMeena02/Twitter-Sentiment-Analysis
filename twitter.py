#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv('train_set for twitter.csv')
test=pd.read_csv('test_set for twitter.csv') 
#combining train and test set    
train['source']='train'
test['source']='test'     
data=pd.concat([train,test],ignore_index=True,sort=True)

#cleaning of text
import nltk
#nltk.download('stopwords')  (if you have not downloaded the stopwords list)
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#removing twitter users
def remove_pattern(input_txt,pattern):
        r=re.findall(pattern,input_txt)
        for i in r:
            input_txt=re.sub(i,'',input_txt)
        return input_txt
data['tidy_tweet']=np.vectorize(remove_pattern)(data['tweet'],"@[\w]*")
#removing puntuations and numbers and all
data['tidy_tweet']=data['tidy_tweet'].str.replace("[^a-zA-Z#]",' ')
#removing short words
data['tidy_tweet']=data['tidy_tweet'].apply(lambda x:' '.join([w for w in x.split() if len(w)>3]))  
#tokenization
tokenized_tweet=data['tidy_tweet'].apply(lambda x: x.split())

#stemming
stemmer=  PorterStemmer()
tokenized_tweet=tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x])

#now stitch these token back together
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i]=' '.join(tokenized_tweet[i])
data['tidy_tweet']=tokenized_tweet

#wordcloud for positive comments
from wordcloud import WordCloud
normal_words=' '.join([text for text in data['tidy_tweet'][data['label']==0]])
wordcloud= WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(normal_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#wordcloud for negative comment
from wordcloud import WordCloud
negative_words=' '.join([text for text in data['tidy_tweet'][data['label']==1]])
wordcloud= WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(negative_words)
plt.figure(figsize=(12,9))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#hashtags are important
#function to collect hashtags
def hashtag_extract(x):
    hashtags=[]
    #loop over the words in tweet
    for i in x:
        ht=re.findall("#(\w+)",i)
        hashtags.append(ht)
    return hashtags

#now extract hashtags from positive and negative comments
ht_normal=hashtag_extract(data['tidy_tweet'][data['label'] == 0])
ht_negative=hashtag_extract(data['tidy_tweet'][data['label']==1])
ht_normal=sum(ht_normal,[])
ht_negative=sum(ht_negative,[])

#plot hashtags on positive ones
a = nltk.FreqDist(ht_normal)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

#plot hashtags on negative ones
b = nltk.FreqDist(ht_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


#TF-IDF Features
#this is different from bow as it takes in to account not just the occurence of  a word in
#single document but in the entire corpus
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(data['tidy_tweet'])   # TF-IDF feature matrix


data.drop(['source','tweet'],axis=1,inplace=True)

#done with all pre model stages
#Building random forest model using Bag-of-Words features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

xtrain_tfidf, xvalid_tfidf, y_train, y_valid = train_test_split(train_tfidf,train['label'], random_state=42, test_size=0.3)

xtrain_tfidf=train_tfidf[y_train.index]
xvalid_tfidf=train_tfidf[y_valid.index]

classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(xtrain_tfidf, y_train) # training the model

predict_valid = classifier.predict_proba(xvalid_tfidf) # predicting on the validation set
valid_predict_int = predict_valid[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
valid_predict_int = valid_predict_int.astype(np.int)

f1_score(y_valid,valid_predict_int) # calculating f1 score

#prediction on test set
test_pred = classifier.predict_proba(test_tfidf)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['label']]
sample=pd.DataFrame.to_csv(submission,index=None)



