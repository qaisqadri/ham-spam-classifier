# sms/mail spam or ham classification
# NO tf_idf

import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
sms = pd.read_table(url, header=None, names=['label', 'message'])
X = sms.message
y = sms.label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()

X_train_dtm = vect.fit_transform(X_train)

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train_dtm, y_train)

X_test_dtm = vect.transform(X_test)

y_pred = nb.predict(X_test_dtm)

from sklearn import metrics
print("Accuracy : ",metrics.accuracy_score(y_test, y_pred))
print("F1 score : ",metrics.accuracy_score(y_test,y_pred))
    
                                                    

