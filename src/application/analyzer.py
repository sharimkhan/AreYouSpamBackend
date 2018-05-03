import pandas as pd
from sklearn.externals import joblib
df = pd.read_table("/Users/sharimkhan/anaconda/AreYouSpam/extrafiles/results1.txt",sep=",")
df = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
df.head()
print(len(df))
df[df.columns[0]] = df[df.columns[0]]
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
print(count_vector.get_params)

#count_vector.fit(documents)
#count_vector.get_feature_names()
#doc_array = count_vector.transform(documents).toarray()
#frequency_matrix = pd.DataFrame(doc_array)
#frequency_matrix.columns = count_vector.get_feature_names()

df.columns = ['label', 'sms_message']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], df['label'], random_state=1)

from sklearn import svm
print("SVM")
clf = svm.SVC(probability=True, C=1000)
clf.fit(training_data,y_train)
svmpredictions = clf.predict(testing_data)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test,svmpredictions )))
print('Precision score: ', format(precision_score(y_test,svmpredictions,average='micro' )))
print('Recall score: ', format(recall_score(y_test,svmpredictions,average='micro' )))
print('F1 score: ', format(f1_score(y_test,svmpredictions,average='micro')))


'''
print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                     solver='adam', verbose=10,  random_state=21,tol=0.000000001)
clf.fit(training_data,y_train)
predictions = clf.predict(testing_data)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('Precision score: ', format(precision_score(y_test,predictions,average='micro')))
print('Recall score: ', format(recall_score(y_test,predictions,average='micro')))
print('F1 score: ', format(f1_score(y_test,predictions,average='micro')))



from sklearn.naive_bayes import BernoulliNB
naive_bayes = BernoulliNB()
naive_bayes.fit(training_data,y_train)
predictions = naive_bayes.predict(testing_data)
print("Naive Bayes")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('Precision score: ', format(precision_score(y_test,predictions,average='micro')))
print('Recall score: ', format(recall_score(y_test,predictions,average='micro')))
print('F1 score: ', format(f1_score(y_test,predictions,average='micro')))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(1, weights = 'uniform')
neigh.fit(training_data,y_train)
predictions = neigh.predict(testing_data)
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('Precision score: ', format(precision_score(y_test,predictions,average='micro')))
print('Recall score: ', format(recall_score(y_test,predictions,average='micro')))
print('F1 score: ', format(f1_score(y_test,predictions,average='micro')))

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data,y_train)
predictions = naive_bayes.predict(testing_data)
print("Naive Bayes")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('Precision score: ', format(precision_score(y_test,predictions,average='micro')))
print('Recall score: ', format(recall_score(y_test,predictions,average='micro')))
print('F1 score: ', format(f1_score(y_test,predictions,average='micro')))
from sklearn import svm
print("SVM")
clf = svm.SVC(probability=True, C=1000)
clf.fit(training_data,y_train)
svmpredictions = clf.predict(testing_data)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test,svmpredictions )))
print('Precision score: ', format(precision_score(y_test,svmpredictions,average='micro' )))
print('Recall score: ', format(recall_score(y_test,svmpredictions,average='micro' )))
print('F1 score: ', format(f1_score(y_test,svmpredictions,average='micro')))


print("Logistic Regression")
from sklearn import linear_model
regr = linear_model.LogisticRegression(random_state=0, solver='sag')
regr.fit(training_data,y_train)
predictions = regr.predict(testing_data)
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('Precision score: ', format(precision_score(y_test,predictions,average='micro')))
print('Recall score: ', format(recall_score(y_test,predictions,average='micro')))
print('F1 score: ', format(f1_score(y_test,predictions,average='micro')))

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data,y_train)
predictions = naive_bayes.predict(testing_data)
print("Naive Bayes")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('Precision score: ', format(precision_score(y_test,predictions,average='micro')))
print('Recall score: ', format(recall_score(y_test,predictions,average='micro')))
print('F1 score: ', format(f1_score(y_test,predictions,average='micro')))

from sklearn.ensemble import RandomForestClassifier
print("Random Forest")
clf = RandomForestClassifier(n_jobs=-1, random_state=0, criterion='gini')
clf.fit(training_data,y_train)
predictions = clf.predict(testing_data)
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('Precision score: ', format(precision_score(y_test,predictions,average='micro')))
print('Recall score: ', format(recall_score(y_test,predictions,average='micro')))
print('F1 score: ', format(f1_score(y_test,predictions,average='micro')))
print("Tree")
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='gini', random_state=0)
clf.fit(training_data,y_train)
predictions = clf.predict(testing_data)
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('Precision score: ', format(precision_score(y_test,predictions,average='micro')))
print('Recall score: ', format(recall_score(y_test,predictions,average='micro')))
print('F1 score: ', format(f1_score(y_test,predictions,average='micro')))





from sklearn import svm
print("SVM")
clf = svm.SVC(probability=True, C=1000)
clf.fit(training_data,y_train)
svmpredictions = clf.predict(testing_data)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test,svmpredictions )))
print('Precision score: ', format(precision_score(y_test,svmpredictions,average='micro' )))
print('Recall score: ', format(recall_score(y_test,svmpredictions,average='micro' )))
print('F1 score: ', format(f1_score(y_test,svmpredictions,average='micro')))

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                     solver='adam', verbose=10,  random_state=21,tol=0.000000001)
clf.fit(training_data,y_train)
predictions = clf.predict(testing_data)
print('Accuracy score: ', format(accuracy_score(y_test,svmpredictions )))
print('Precision score: ', format(precision_score(y_test,svmpredictions,average='micro' )))
print('Recall score: ', format(recall_score(y_test,svmpredictions,average='micro' )))
print('F1 score: ', format(f1_score(y_test,svmpredictions,average='micro')))

print("SVM")
clf = svm.SVC(probability=True, C=1200)
clf.fit(training_data,y_train)
svmpredictions = clf.predict(testing_data)
#print(X_test[y_test != svmpredictions])
#joblib.dump(clf,"mymodel.joblib.pkl",compress=9)
#joblib.dump(count_vector,"vector_pkl",compress=9)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test,svmpredictions )))
print('Precision score: ', format(precision_score(y_test,svmpredictions,average='micro' )))
print('Recall score: ', format(recall_score(y_test,svmpredictions,average='micro' )))
print('F1 score: ', format(f1_score(y_test,svmpredictions,average='micro')))

from sklearn.ensemble import AdaBoostClassifier


clf = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1,
                         random_state=0)
clf.fit(training_data,y_train)
predictions = clf.predict(testing_data)

print("AdaBoostClassifier")
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('Precision score: ', format(precision_score(y_test,predictions,average='micro')))
print('Recall score: ', format(recall_score(y_test,predictions,average='micro')))
print('F1 score: ', format(f1_score(y_test,predictions,average='micro')))
print("Logistic Regression")
from sklearn import linear_model
regr = linear_model.LogisticRegression(random_state=0, solver='sag')
regr.fit(training_data,y_train)
predictions = regr.predict(testing_data)
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('Precision score: ', format(precision_score(y_test,predictions,average='micro')))
print('Recall score: ', format(recall_score(y_test,predictions,average='micro')))
print('F1 score: ', format(f1_score(y_test,predictions,average='micro')))
print("KNN")
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(3, weights = 'uniform')
neigh.fit(training_data,y_train)
predictions = neigh.predict(testing_data)
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('Precision score: ', format(precision_score(y_test,predictions,average='micro')))
print('Recall score: ', format(recall_score(y_test,predictions,average='micro')))
print('F1 score: ', format(f1_score(y_test,predictions,average='micro')))
from sklearn.ensemble import RandomForestClassifier
print("Random Forest")
clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(training_data,y_train)
predictions = clf.predict(testing_data)
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('Precision score: ', format(precision_score(y_test,predictions,average='micro')))
print('Recall score: ', format(recall_score(y_test,predictions,average='micro')))
print('F1 score: ', format(f1_score(y_test,predictions,average='micro')))
print("Tree")
from sklearn import tree
clf = tree.DecisionTreeClassifier( random_state=0)
clf.fit(training_data,y_train)
predictions = clf.predict(testing_data)
print('Accuracy score: ', format(accuracy_score(y_test,predictions)))
print('Precision score: ', format(precision_score(y_test,predictions,average='micro')))
print('Recall score: ', format(recall_score(y_test,predictions,average='micro')))
print('F1 score: ', format(f1_score(y_test,predictions,average='micro')))
'''
