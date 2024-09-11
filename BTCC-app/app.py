from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
import joblib
from sklearn.model_selection import train_test_split

import seaborn as sns
import numpy as np
import pandas as pd
import string
import re
import nltk

import subprocess
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer
import nltk
nltk.download('punkt')

#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize



app= Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    df=pd.read_csv("banglishAmad.csv")
    # df_data=df[["text","toxic"]]

    def filt_URLS(text):
        return re.sub(r'https:?//\S+|www\.',"",text)
    def filt_emails(text):
        return re.sub(r'^[a-zA-Z0-9_-]+[@][a-zA-Z]+[.].+',"",text)
    def filt_html(text):
        return re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});',"",text)
    def filt_punctuations(text):
        k = string.punctuation
        return text.translate(str.maketrans(' ',' ',k) )
    def filt_emojis(text):
        emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF""]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    def filt_non_ascii(text):
        stripped = (c for c in text if 0 < ord(c) < 127)
        return ''.join(stripped)

    def filt_chars(text):
        return re.sub(r'\n'," ",text)
    
    #here, we have applied those functions
    df['text'] = df['text'].apply(lambda x:x.lower())
    #df['text'] = df['text'].apply(lambda x:ctrs.fix(x))
    df['text'] = df['text'].apply(lambda x:filt_URLS(x))
    df['text'] = df['text'].apply(lambda x:filt_emails(x))
    df['text'] = df['text'].apply(lambda x:filt_html(x))
    df['text'] = df['text'].apply(lambda x:filt_punctuations(x))
    df['text'] = df['text'].apply(lambda x:filt_emojis(x))
    df['text'] = df['text'].apply(lambda x:filt_non_ascii(x))
    df['text'] = df['text'].apply(lambda x:filt_chars(x))



    #apply "random over sampling balancing technique"
    #class count
    class_count_0, class_count_1 = df['toxic'].value_counts()
    # Separate classes
    class_0 = df[df['toxic'] == 0]
    class_1 = df[df['toxic'] == 1]
   
    class_1_over = class_1.sample(class_count_0, replace=True)
    df2 = pd.concat([class_1_over, class_0], axis=0)


    from nltk.tokenize import sent_tokenize, word_tokenize



    # Tokenize sentences
    #sentences = sent_tokenize(df2["text"])

    # Tokenize words in each sentence and create one-hot encoding
    #word_to_index = {}
    #encoded_sentences = []

    #for sentence in sentences:
       # words = word_tokenize(sentence)
       # encoded_sentence = [word_to_index.setdefault(word, len(word_to_index)) for word in words]
       # encoded_sentences.append(encoded_sentence)

    #df_x=encoded_sentences


    df_x= df2["text"]
    #token = Tokenizer(num_words=150, oov_token="<oov>")
    #token= df2['text'].apply(word_tokenize)
    #token.fit_on_texts(df2['text'])
    #word_index=token.word_index
    #sequence=token.texts_to_sequences(df2['text'])
    #padded=pad_sequences(sequence, padding='post', truncating='post')
    #df_x=token.tolist()
    #df_x=df_x.toarray()

    #df_y=df2.toxic.tolist()
    #df_y=df_y.toarray()

    #corpus =df_x
    df_y=df2["toxic"]
    test_corpus = df_x
    
    tfidf_test=TfidfVectorizer()
    emb = tfidf_test.fit_transform(test_corpus)
    X_train, X_test, y_train, y_test= train_test_split(emb, df_y, test_size=0.01, random_state=42)

    from sklearn.svm import SVC
    svc_model = SVC(kernel='poly')
    svc_model.fit(X_train, y_train)
    from sklearn.metrics import accuracy_score
    svc_model.score(X_test,y_test)
     
    #y_model1=open('svm_Btcc.pkl',"rb")
    #svc_model=joblib.load(y_model1)


    if request.method== 'POST':
        comment=request.form['comment']
        data=[comment]
        
        #token1=word_tokenize(data)
       #tok=word_to_index.setdefault(token1, len(word_to_index))
        #token1=token1.fit_on_texts(data)
        #sequence1=token1.texts_to_sequences(data)
        #padded1=pad_sequences(sequence1, padding='post', truncating='post')
        #data=token1.tolist()
        #data1=data.toarray()

        vect=tfidf_test.transform(data).toarray()
        
        my_prediction =svc_model.predict(vect)



    return render_template('results.html', prediction=my_prediction)

if __name__ =='__main__':
    app.run(debug=True)

   
