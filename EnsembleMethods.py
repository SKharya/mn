from sklearn.model_selection import train_test_split

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from mlxtend.classifier import StackingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn

dataset = pd.read_csv('D:/Niranjan/Data Science/Semester 2/Python & ML data sets/redWineQuality.csv', sep=';')
X, y = dataset.iloc[:, 0:11].values, dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size = 0.3)

model1 = DecisionTreeClassifier(criterion='gini')
model2 = KNeighborsClassifier(n_neighbors=3)
model3 = LogisticRegression(penalty='l1')

bag1 = BaggingClassifier(base_estimator=model1, n_estimators=10)
bag2 = BaggingClassifier(base_estimator=model2, n_estimators=10)
bag3 = BaggingClassifier(base_estimator=model3, n_estimators=10)

boost1 = AdaBoostClassifier(base_estimator=model1, n_estimators=10)
boost3 = AdaBoostClassifier(base_estimator=model3, n_estimators=10)

stack = StackingClassifier(classifiers=[model1, model2], meta_classifier=model3)


label = ['Decision Tree', 'Logistic Reg', 'K-NN', 'Bagging Tree', 'Bagging K-NN', 'Bagging Logistic Reg']
clf_list = [model1, model2, model3, bag1, bag2, bag3]

sn.heatmap(pd.DataFrame(confusion_matrix(y_test, model3.predict(X_test))), annot=True)

#Bagging
for clf, mLabel in zip(clf_list, label):
    clf.fit(X_train, y_train)
    print(clf.score(X, y), 'Training Score for: ', mLabel)
    pred = clf.predict(X_test)
    print('Testing Score for: ', mLabel, accuracy_score(y_test, pred))
    print('\n')

#Boosting
label = ['Decision Tree', 'Logistic Reg', 'K-NN', 'Boosting Tree', 'Boosting Logistic Reg']
clf_list = [model1, model2, model3, boost1, boost3]
for clf, mLabel in zip(clf_list, label):
    clf.fit(X_train, y_train)
    print(clf.score(X, y), 'Training Score for: ', mLabel)
    pred = clf.predict(X_test)
    print('Testing Score for: ', mLabel, accuracy_score(y_test, pred))
    print('\n')

#Stacking
label = ['Decision Tree', 'Logistic Reg', 'K-NN', 'Stacking Classifier']
clf_list = [model1, model2, model3, stack]

for clf, mLabel in zip(clf_list, label):
    clf.fit(X_train, y_train)
    print(clf.score(X, y), 'Training Score for: ', mLabel)
    pred = clf.predict(X_test)
    print('Testing Score for: ', mLabel, accuracy_score(y_test, pred))
    print('\n')

#The data is of high variance as we can see the difference between the training and test data is very high
#But we can see that the bagging, boosting and stacking emsemble methods reduce the differnce in the error
#thus making the model less overfitting
