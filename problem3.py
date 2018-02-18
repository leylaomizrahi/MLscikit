# Import Necessary 
import csv 
from sklearn.svm import SVC
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn import datasets, svm
from sklearn import cross_validation
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble

x = []
y = []

# Retrive Data from input                                                                             
file_reader = csv.reader(open(sys.argv[1], 'rU'), delimiter=',')
file_writer = csv.writer(open(sys.argv[2], 'wb'), delimiter=',')

name = ['svm_linear','svm_polynomial','svm_rbf','logistic','knn','decision_tree','random_forest']

i = 0 
for row in file_reader:
     if i == 0: 
          i = i + 1
          continue
     y.append( float(row[2]))
     x.append((float(row[0]), float(row[1])))
     i = i +1 

#Create Stratified Samples
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.4, stratify= y)


#set up variables
n_neighbors = []
leaf_size = []
for i in range(1,51):
     n_neighbors.append(i)
for i in range (1,13):
     leaf_size.append(i * 5)
max_depth = n_neighbors

#set up parameters
tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}, {'C': [0.1,1,3], 'gamma': [0.1, 1], 'kernel': ['poly']},{'C': [0.1,0.5, 1,5,10,50,100], 'gamma': [0.1,0.5,1,3,6,10], 'kernel': ['rbf']},{'C':[0.1,0.5, 1,5,10,50,100]},{'n_neighbors':n_neighbors, 'leaf_size':leaf_size},{'min_samples_split':[2,3,4,5,6,7,8,9,10], 'max_depth':max_depth}, {'min_samples_split':[2,3,4,5,6,7,8,9,10], 'max_depth':max_depth}]

a = svm.SVC() 
b = svm.SVC()
c = svm.SVC() 
d = linear_model.LogisticRegression() 
e = KNeighborsClassifier()
f = tree.DecisionTreeClassifier()
g = ensemble.RandomForestClassifier()

est = [ a,b,c,d,e,f,g]

best_parameters = [] 
params = []
best = 0 
train_scores = []
test_scores = [] 
#for i in range(len(tuned_parameters)):
for i in range(len(tuned_parameters)):
     clf = GridSearchCV(estimator=est[i],param_grid=tuned_parameters[i], n_jobs=-1, cv = 5)
     clf.fit(X_train, y_train)
     #file_writer.writerow([name[i],clf.best_score_, clf.score(X_test, y_test)])
     print name[i]
     train_scores.append(clf.best_score_)
     print "train: ",clf.best_score_     
     print "test: ",clf.score(X_test, y_test)
     print clf.best_params_

linear = svm.SVC()
linear.set_params(C = 1)
linear.fit(X_test, y_test)
linear_score =  linear.score(X_test,y_test)
test_scores.append(linear_score)

linear = svm.SVC()
linear.set_params(gamma = 0.1, C = 0.1)
linear.fit(X_test, y_test)
linear_score =  linear.score(X_test,y_test)

test_scores.append(linear_score)

linear = svm.SVC()
linear.set_params(C = 50, gamma = 1)
linear.fit(X_test, y_test)
linear_score =  linear.score(X_test,y_test)

test_scores.append(linear_score)
linear = linear_model.LogisticRegression(C = 0.1)
linear.fit(X_test, y_test)
linear_score =  linear.score(X_test,y_test)

test_scores.append(linear_score)

linear = KNeighborsClassifier(n_neighbors =  3,leaf_size =  5)
linear.fit(X_test, y_test)
linear_score =  linear.score(X_test,y_test)

test_scores.append(linear_score)

linear =  tree.DecisionTreeClassifier(min_samples_split=  2, max_depth =  5)
linear.fit(X_test, y_test)
linear_score =  linear.score(X_test,y_test)

test_scores.append(linear_score)

linear =  ensemble.RandomForestClassifier(min_samples_split=  7, max_depth =  9)
linear.fit(X_test, y_test)
linear_score =  linear.score(X_test,y_test)

test_scores.append(linear_score)



print train_scores
print test_scores 
