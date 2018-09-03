# sms/mail spam or ham classification
# uses tfidf

import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
sms = pd.read_table(url, header=None, names=['label', 'message'])
X = sms.message
y = sms.label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

vect = CountVectorizer()

X_train_dtm = vect.fit_transform(X_train)

# calculate tf_idf for training

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_dtm)


#print(X_train_dtm.toarray())
#print(X_train_tfidf.toarray())


from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb2=MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb2.fit(X_train_dtm, y_train)
# calculate tf_idf for testing

X_test_dtm = vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_dtm)

y_pred = nb.predict(X_test_tfidf)
y_pred2 = nb2.predict(X_test_dtm)

test2=["hello there, congratulations for winning a lottery worth 1 million dollars, click on the link below to claim. ","Hello sir! i would like to imform you regarding my text classification task. i hv completed it"]
test2_dtm=vect.transform(test2)

print(nb.predict(test2_dtm))

from sklearn import metrics
print("using tf_idf")
a1=metrics.accuracy_score(y_test, y_pred)
print("Accuracy : ",a1)
print("F1 score : ",metrics.accuracy_score(y_test,y_pred))
print("without using td_idf")
a2=metrics.accuracy_score(y_test, y_pred2)
print("Accuracy : ",a2)
print("F1 score : ",metrics.accuracy_score(y_test,y_pred2))
    
print("difference : ")

print(a2 - a1)
                                                    

