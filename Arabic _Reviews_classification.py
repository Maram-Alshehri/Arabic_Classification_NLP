#----------------------------------------------------------------------------------------------
#----------------------------------------libraray used ----------------------------------------
#----------------------------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split

# for preprocessing text
import re 
import string
import nltk
from nltk.corpus import stopwords

# for Bag of words
from sklearn.feature_extraction.text import CountVectorizer

# for classifier model
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import  classification_report
from sklearn.svm import SVC


#----------------------------------------------------------------------------------------------
#----------------------------------------Read dataset -----------------------------------------
#----------------------------------------------------------------------------------------------

dataset = pd.read_excel('restuarantDataSet.xlsx', encoding ='utf-8-sig')
print(dataset)
#----------------------------------------------------------------------------------------------
#----------------------------------------Cleaning dataset -------------------------------------
#----------------------------------------------------------------------------------------------

punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
arabic_stopwords = stopwords.words('arabic')

def clean_text(text):
 
# remove punctuation
    text = text.translate(str.maketrans('','', punctuations))
# remove stop word
    text = ' '.join([word for word in text.split() if word not in arabic_stopwords])
#remove line space and space    
    text = text.lstrip()
    text = text.strip()
  
    return text



#----------------------------------------------------------------------------------------------
#--------------------------------------Create Bag of words ------------------------------------
#----------------------------------------------------------------------------------------------

count_vector = CountVectorizer()  
X = count_vector.fit_transform(dataset['review ']).toarray()

d = pd.DataFrame(X,columns=count_vector.get_feature_names())


#----------------------------------------------------------------------------------------------
#------------------------------Split dataset into train and test ------------------------------
#----------------------------------------------------------------------------------------------

y = dataset['class']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.3, random_state=0)

#----------------------------------------------------------------------------------------------
#------------------------------Classification Algorithm  --------------------------------------
#----------------------------------------------------------------------------------------------

#------------------------------Naive Bayes Algorithm  -----------------------------------------
#GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnbpredictions = gnb.predict(X_test)
print("----------------------- GaussianNB--------------------- \n",classification_report(y_test, gnbpredictions))

#MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
mnbpredictions = mnb.predict(X_test)
print("----------------------- MultinomialNB-------------------- \n",classification_report(y_test, mnbpredictions))

#BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
bnbpredictions = bnb.predict(X_test)
print("----------------------- BernoulliNB -------------------\n",classification_report(y_test, bnbpredictions))

#------------------------------  SVM Algorithm  -----------------------------------------------

from sklearn.model_selection import GridSearchCV 
param_grid = {'C': [1, 5, 10, 50],  

              'gamma': [0.0001, 0.0005, 0.001, 0.005], 

              'kernel': ['rbf','linear','poly','sigmoid']}  

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
grid.fit(X_train, y_train)


print('#the optimal parameter ' ,grid.best_params_)

y_predictions = grid.predict(X_test) 
print("----------------------- SVM -----------------------\n" ,classification_report(y_test, y_predictions))









