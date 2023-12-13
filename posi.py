import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,Embedding,Flatten
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
#import np_utils
from sklearn.preprocessing import LabelEncoder
import sys
import time
import datetime
import warnings
import tensorflow.keras
import re,pymsgbox

warnings.filterwarnings("ignore")

dataFrame=pd.read_csv('train.csv', encoding='ISO-8859-1') #error_bad_lines=False
x=dataFrame.values[:,2]
y=dataFrame.values[:,1]

# ----------------------------Preprocessing Text Data---------------------------------------

stop_words = set(stopwords.words('english')) 
new_stop_words=set(stop_words)

# adding woudlnt type of words into stopwords list
for s in stop_words:
	new_stop_words.add(s.replace('\'',''))
	pass
	
stop_words=new_stop_words
print("Excluding stopwords ...")

# removing @ from default base filter, to remove that whole word, which might be considered as user or page name
base_filters='\n\t!"#$%&()*+,-./:;<=>?[\]^_`{|}~ '

word_sequences=[]

for i in x:
	i=str(i)
	i=i.replace('\'', '')
	newlist = [x for x in text_to_word_sequence(i,filters=base_filters, lower=True) if not x.startswith("@")]
	filtered_sentence = [w for w in newlist if not w in stop_words] 
	word_sequences.append(filtered_sentence)
	pass


tokenizer = Tokenizer()
tokenizer.fit_on_texts(word_sequences)
word_indices = tokenizer.texts_to_sequences(word_sequences)
word_index = tokenizer.word_index
print("Tokenized to Word indices as ")
print(np.array(word_indices).shape)

#padding word_indices
MAX_SEQUENCE_LENGTH = 20
x_data=pad_sequences(word_indices,maxlen=MAX_SEQUENCE_LENGTH)
print("After padding data")
print(x_data.shape)

from tensorflow.keras.utils import to_categorical
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
le_name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_),label_encoder.classes_))
print("Label Encoding Classes as ")
print(le_name_mapping)

y_data=to_categorical(integer_encoded)
print("One Hot Encoded class shape ")
print(y_data.shape)

print("Finished Preprocessing data ...")
print("x_data shape : ",x_data.shape)
print("y_data shape : ",y_data.shape)

# spliting data into training, testing set
print("spliting data into training, testing set")
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data)

#Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1])
# create the model
embedding_vector_length = 50
top_words=60000
max_review_length = 20
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Flatten())
#model.add(Dense(500,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

batch_size = 74991
num_epochs = 1
history=model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=num_epochs)

dataFrame=pd.read_csv('train.csv', encoding='ISO-8859-1') #error_bad_lines=False
x=dataFrame.values[:,2]
print(x)
y=dataFrame.values[:,1]
print(y)
# ----------------------------Preprocessing Text Data---------------------------------------
def preprocess_tweet(tweet):
	#========Preprocess the text in a single tweet 
	#convert the tweet to lower case
	tweet.lower()
	#=========convert all urls to sting "URL"
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
	#=========convert all @username to "AT_USER"
	tweet = re.sub('@[^\s]+','AT_USER', tweet)
	#=========correct all multiple white spaces to a single white space
	tweet = re.sub('[\s]+', ' ', tweet)
	#=========convert "#topic" to just "topic"
	tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
	return tweet

def feature_extraction(data, method = "tfidf"):	
	#===========methods of feature extractions: "tfidf" and "doc2vec"
	if method == "tfidf":
		from sklearn.feature_extraction.text import TfidfVectorizer
		tfv=TfidfVectorizer(sublinear_tf=True, stop_words = "english")
		features=tfv.fit_transform(data)
	else:
		return "Incorrect inputs"
	return features

        
dataFrame['SentimentText'] = dataFrame['SentimentText'].apply(preprocess_tweet)
data_tr = np.array(dataFrame.SentimentText)
label_tr = np.array(dataFrame.Sentiment)
features_tr = feature_extraction(data_tr, method = "tfidf")

# spliting data into training, testing set
print("spliting data into training, testing set")
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=0)
print(x_train)
print(y_train)
print(x_test)
print(y_test)


############new input classification############
a=pymsgbox.prompt('enter the command', title='This reference dates this example.')
a_p=preprocess_tweet(a)
a_pre=np.array([a_p])
f1=pd.read_csv('train.csv', encoding='ISO-8859-1') #error_bad_lines=False
da=f1.values[:,2]
cv=f1.values[:,1]
a_pre=np.array(da)
a_fet=feature_extraction(a_pre,method = "tfidf")

from textblob import TextBlob
c=TextBlob(a)
d=c.sentiment.polarity
if d>0.5:
        pymsgbox.alert("positive", 'This Command was!')
        
        dataFrame=pd.read_csv('positive_dataset.csv', encoding='utf-8') #error_bad_lines=False
        x=dataFrame.values[:,0]
        y=dataFrame.values[:,1]

        import nltk
        nltk.download('stopwords')
        # ----------------------------Preprocessing Text Data---------------------------------------

        stop_words = set(stopwords.words('english')) 
        new_stop_words=set(stop_words)

        # adding woudlnt type of words into stopwords list
        for s in stop_words:
                new_stop_words.add(s.replace('\'',''))
                pass
                
        stop_words=new_stop_words
        print("Excluding stopwords ...")

        # removing @ from default base filter, to remove that whole word, which might be considered as user or page name
        base_filters='\n\t!"#$%&()*+,-./:;<=>?[\]^_`{|}~ '

        word_sequences=[]

        for i in x:
                i=str(i)
                i=i.replace('\'', '')
                newlist = [x for x in text_to_word_sequence(i,filters=base_filters, lower=True) if not x.startswith("@")]
                filtered_sentence = [w for w in newlist if not w in stop_words] 
                word_sequences.append(filtered_sentence)
                pass

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(word_sequences)
        word_indices = tokenizer.texts_to_sequences(word_sequences)
        word_index = tokenizer.word_index
        print("Tokenized to Word indices as ")
        print(np.array(word_indices).shape)

        #padding word_indices
        MAX_SEQUENCE_LENGTH = 20
        x_data=pad_sequences(word_indices,maxlen=MAX_SEQUENCE_LENGTH)
        print("After padding data")
        print(x_data.shape)

        from tensorflow.keras.utils import to_categorical
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(y)
        le_name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_),label_encoder.classes_))
        print("Label Encoding Classes as ")
        print(le_name_mapping)

        y_data=to_categorical(integer_encoded)
        print("One Hot Encoded class shape ")
        print(y_data.shape)


        print("Finished Preprocessing data ...")
        print("x_data shape : ",x_data.shape)
        print("y_data shape : ",y_data.shape)

        # spliting data into training, testing set
        print("spliting data into training, testing set")
        x_train,x_test,y_train,y_test=train_test_split(x_data,y_data)


        # create the model
        embedding_vecor_length = 50
        top_words=50000
        max_review_length = 20
        model = Sequential()
        model.add(Embedding(top_words, embedding_vecor_length, input_length=20))
        model.add(LSTM(100))
        model.add(Flatten())
        #model.add(Dense(500,activation='relu'))
        model.add(Dense(300,activation='relu'))
        model.add(Dense(7, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        batch_size = 30000
        num_epochs = 250
        history=model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=num_epochs)
        
        word_sequences=[]

        for i in a:
                i=str(i)
                i=i.replace('\'', '')
                newlist = [x for x in text_to_word_sequence(i,filters=base_filters, lower=True) if not x.startswith("@")]
                filtered_sentence = [w for w in newlist if not w in stop_words] 
                word_sequences.append(filtered_sentence)
                pass

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(word_sequences)
        word_indices = tokenizer.texts_to_sequences(word_sequences)
        word_index = tokenizer.word_index
        print("Tokenized to Word indices as ")
        print(np.array(word_indices).shape)

        #padding word_indices
        MAX_SEQUENCE_LENGTH = 20
        x_data=pad_sequences(word_indices,maxlen=MAX_SEQUENCE_LENGTH)
        print("After padding data")
        print(x_data.shape)
        
        print(model.predict(x_data))
        y_pred_cnn=model.predict(x_data)
        for i in y_pred_cnn:
                        print("\n")
                        print("enthusiasm=",i[0],"fun=",i[1],
                          "happiness=",i[2],"love=",i[3],
                          "neutral=",i[4],"relief=",i[5],
                          "surprice=",i[6])
                        break
        print =("Want to enter more type yes")
        b=pymsgbox.prompt('you want enter one more', title='This reference dates this example.')

        if b!="yes":
                more = False 
                pymsgbox.alert("thankyou", 'This Command was!')


        
        
        
else:
        pymsgbox.alert("negative", 'This Command was!')

        dataFrame=pd.read_csv('text_emotion.csv', encoding='utf-8') #error_bad_lines=False
        x=dataFrame.values[:,0]
        y=dataFrame.values[:,1]

        import nltk
        nltk.download('stopwords')
        # ----------------------------Preprocessing Text Data---------------------------------------

        stop_words = set(stopwords.words('english')) 
        new_stop_words=set(stop_words)

        # adding woudlnt type of words into stopwords list
        for s in stop_words:
                new_stop_words.add(s.replace('\'',''))
                pass
                
        stop_words=new_stop_words
        print("Excluding stopwords ...")

        # removing @ from default base filter, to remove that whole word, which might be considered as user or page name
        base_filters='\n\t!"#$%&()*+,-./:;<=>?[\]^_`{|}~ '

        word_sequences=[]

        for i in x:
                i=str(i)
                i=i.replace('\'', '')
                newlist = [x for x in text_to_word_sequence(i,filters=base_filters, lower=True) if not x.startswith("@")]
                filtered_sentence = [w for w in newlist if not w in stop_words] 
                word_sequences.append(filtered_sentence)
                pass

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(word_sequences)
        word_indices = tokenizer.texts_to_sequences(word_sequences)
        word_index = tokenizer.word_index
        print("Tokenized to Word indices as ")
        print(np.array(word_indices).shape)

        #padding word_indices
        MAX_SEQUENCE_LENGTH = 20
        x_data=pad_sequences(word_indices,maxlen=MAX_SEQUENCE_LENGTH)
        print("After padding data")
        print(x_data.shape)

        from tensorflow.keras.utils import to_categorical
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(y)
        le_name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_),label_encoder.classes_))
        print("Label Encoding Classes as ")
        print(le_name_mapping)

        y_data=to_categorical(integer_encoded)
        print("One Hot Encoded class shape ")
        print(y_data.shape)


        print("Finished Preprocessing data ...")
        print("x_data shape : ",x_data.shape)
        print("y_data shape : ",y_data.shape)

        # spliting data into training, testing set
        print("spliting data into training, testing set")
        x_train,x_test,y_train,y_test=train_test_split(x_data,y_data)


        # create the model
        embedding_vecor_length = 50
        top_words=50000
        max_review_length = 20
        model = Sequential()
        model.add(Embedding(top_words, embedding_vecor_length, input_length=20))
        model.add(LSTM(100))
        model.add(Flatten())
        #model.add(Dense(500,activation='relu'))
        model.add(Dense(300,activation='relu'))
        model.add(Dense(6, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        batch_size = 30000
        num_epochs = 250
        history=model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=num_epochs)
        
        word_sequences=[]

        for i in a:
                i=str(i)
                i=i.replace('\'', '')
                newlist = [x for x in text_to_word_sequence(i,filters=base_filters, lower=True) if not x.startswith("@")]
                filtered_sentence = [w for w in newlist if not w in stop_words] 
                word_sequences.append(filtered_sentence)
                pass

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(word_sequences)
        word_indices = tokenizer.texts_to_sequences(word_sequences)
        word_index = tokenizer.word_index
        print("Tokenized to Word indices as ")
        print(np.array(word_indices).shape)

        #padding word_indices
        MAX_SEQUENCE_LENGTH = 20
        x_data=pad_sequences(word_indices,maxlen=MAX_SEQUENCE_LENGTH)
        print("After padding data")
        print(x_data.shape)
        
        print(model.predict(x_data))
        y_pred_cnn=model.predict(x_data)
        for i in y_pred_cnn:
                        print("\n")
                        print("angry=",i[0],"boredom=",i[1],
                          "empty=",i[2],"hate=",i[3],
                          "sadness=",i[4],"worry",i[5],
                          )
                        break
        print =("Want to enter more type yes")
        b=pymsgbox.prompt('you want enter one more', title='This reference dates this example.')

        if b!="yes":
                more = False 
                pymsgbox.alert("thankyou", 'This Command was!')



        print("thank-you")
