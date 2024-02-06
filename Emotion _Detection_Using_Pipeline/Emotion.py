# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import nltk
import re
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB,MultinomialNB,CategoricalNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder,FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.impute import SimpleImputer

# 1.Data Loading and Overview


data = pd.read_csv(r"C:\Users\Ranjith\Downloads\Emotion_classify_Data.csv")
data.head()
#copying the data to anthoer variable to store the original data 
datac = data.copy()

# Overview of the data
datac.head()

datac.info() 

datac.shape

datac.duplicated().sum()

datac.describe()

datac.isnull().sum()


# 2. Exploratory Data Analysis (EDA)  &  Text Preprocessing


# Checking for the differt types of emotions present in given data
datac['Emotion'].value_counts()

# Dividing  the given text  data  into there respective emotions as a single string
anger_emotion = ' '.join(datac[datac['Emotion'] == 'anger']['Comment'])
joy_emotion= ' '.join(datac[datac['Emotion'] == 'joy']['Comment'])
fear_emotion = ' '.join(datac[datac['Emotion'] == 'fear']['Comment'])


# creating wordcloud for each of the emotion.

wc = WordCloud().generate(anger_emotion)
plt.imshow(wc)
plt.show()

wc = WordCloud().generate(joy_emotion)
plt.imshow(wc)
plt.show()

wc = WordCloud().generate(fear_emotion)
plt.imshow(wc)
plt.show()

# dividing the data into feature_variables and class_variable

fv = datac.iloc[:,0]
cv = datac.iloc[:,1]

fv.head()

# "Using the feature variables (fv) and the class variable (cv), the data has been split into training and test sets.

x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,random_state=3,stratify = cv)


# creating a pipeline based on the giving data
 

def lower(x):
    return x.str.lower()
def html(x):
    return x.apply(lambda x:re.sub("<.+?>"," ",x))
def url(x):
    return x.apply(lambda x:re.sub("[.+]?http[s]?://.+? +"," ",x))
def unw(x):
    return x.apply(lambda x:re.sub("[]\?:()*\-.!,@#$%^&0-9]"," ",x))
def preprocess_text(x):
    stp = stopwords.words('english')
    def advpp(x,stemm):
        wl = WordNetLemmatizer()
        l = []
        for word in word_tokenize(x):
            if word in stp:
                pass
            else:
                if stemm == 'r':
                    l.append(wl.lemmatize(word,pos = 'v'))
                else:
                    l.append(word)      
        return ' '.join(l)
    return x.apply(advpp,args=('r'))


pre_pro_pi = Pipeline([("lower",FunctionTransformer(lower)),("html",FunctionTransformer(html)),("url",FunctionTransformer(url)),
                       ("unw",FunctionTransformer(unw)),("preprocess_text",FunctionTransformer(preprocess_text))])


final_pip = Pipeline([("pre_process",pre_pro_pi),("vectorizer",CountVectorizer())])
final_pip.fit_transform(x_train)



# creating a pickle file (.pkl) for the final pipeline

import pickle
pickle.dump(final_pip,open(r"C:\Users\Ranjith\OneDrive\Pictures\Pipeline\finalp1_emotion.pkl","wb"))




# 3. Model Training

# creating a model based on the data and pickel file(.pkl) for using it in testing purpose
mb = MultinomialNB()
model = mb.fit(final_pip.fit_transform(x_train),y_train)
pickle.dump(model,open(r"C:\Users\Ranjith\OneDrive\Pictures\Pipeline\twitter_emotion.pkl","wb"))



# 4. Streamlit App Integration

# creating a streamlit application 

import streamlit as st
st.title("Emotion Detection")

finalf = pickle.load(open(r"C:\Users\Ranjith\OneDrive\Pictures\Pipeline\finalp1_emotion.pkl","rb"))
q = st.text_input("Enter the text")
Query = finalf.transform(pd.Series(q))
modelf = pickle.load(open(r"C:\Users\Ranjith\OneDrive\Pictures\Pipeline\emotion_model.pkl","rb"))
predict = modelf.predict(Query)
if st.button('Submit'):
    st.header(predict)



