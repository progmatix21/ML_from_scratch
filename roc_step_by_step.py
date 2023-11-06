#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:02:19 2023

@author: projects
"""

#%% Import modules

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression

#%% Utility functions
def get_auc(X,Y):
    # get area-under-curve for curve (X,Y)
    # both X and Y should be limited by limits [0,1]
    # Use trapezium rule: area of a sliver is (x2-x1)*(y2+y1)/2
    # Area under curve is sum of all the slivers bounded by right limits
    # X1 to Xn
    X_diff = np.diff(X)
    Y_sum = (Y + np.roll(Y,1))[1:]
    return np.sum(X_diff * Y_sum / 2)    

#%% Create dataset

X,y = make_classification(n_samples=10000,n_features=2,n_informative=2,
                          n_redundant=0,n_repeated=0,class_sep=1.0,
                          flip_y=0.1, # anamolous labels
                          weights=[0.9,0.1], # Create class imbalance
                          random_state=21) 

sns.scatterplot(x=X[:,0],y=X[:,1],hue=y)
plt.title("Generated Dataset")
plt.xlabel("$X_0$")
plt.ylabel("$X_1$")

plt.show()

#%% Train classifier

clf = LogisticRegression()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4)

clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)

#Get probabilities for class 1
clf_probs = clf.predict_proba(X_test)[:,1]

#%% Get FPR and TPR values

# Thresholds to use for plotting
thresh = np.linspace(0.95,0.00,num=20,endpoint=True)

# For each threshold, get predictions, a confusion matrix and
# FPR and TPR values. These FPR, TPR tuples are our points for
# the ROC plot.

fpr = []
tpr = []  # List to hold fpr, tpr 
precision = []
acc = []

for t in thresh:
    clf_preds = (clf_probs > t).astype(int)
    tn,fp,fn,tp = confusion_matrix(y_test,clf_preds).ravel()
    fpr.append(fp/(fp+tn))
    precision.append(tp/(tp+fp))
    tpr.append(tp/(tp+fn)) 
    acc.append((tp+tn)/(tp+tn+fp+fn))
#%% Plot ROC curve

plt.plot(fpr,tpr,label=f'ROC AUC:{get_auc(fpr,tpr):0.2f}')
plt.scatter(fpr,tpr,s=thresh*200,alpha=0.4) # superimpose markers scaled on prob threshold
plt.scatter([fpr[9]],[tpr[9]],c='r',s=100)  # red dot for 0.5 threshold

plt.plot(tpr,precision, label=f'PR AUC:{get_auc(tpr,precision):0.2f}')
plt.scatter(tpr,precision,s=thresh*200,alpha=0.4)
plt.scatter([tpr[9]],[precision[9]],c='r',s=100)  # red dot for 0.5 threshold

plt.plot(thresh,acc,'c^--', markersize=4, label='acc vs thresh')

plt.text(0.3,0.4,'Marker sizes denote prob. thresholds')
plt.text(0.3,0.35,'Red marker denotes 0.5 prob. threshold')
plt.plot((0.0,1.0),(0.0,1.0),'k:',label='ROC dummy')

plt.title(f'ROC & PR Curves accuracy:{accuracy:0.2f}')
plt.xlabel('FPR(ROC) or Recall(PR)')
plt.ylabel('TPR(ROC) or Precision(PR)')
plt.grid()
plt.legend(loc='lower right')
plt.show()

 