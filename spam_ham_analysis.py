# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:35:27 2021

@author: Chetan
"""
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

df = pd.read_csv("spam_ham.csv")
df.isnull().sum()

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(df['label'])
y=y.iloc[:,1].values


df["label"].value_counts()

## Imbalanced data handling
from imblearn.combine import SMOTETomek

# Implementing Oversampling for Handling Imbalanced 
smk = SMOTETomek(random_state=42)
X,y=smk.fit_resample(X,y)

X.shape,y.shape

# Train Test Split
from sklearn.model_selection import train_test_split,GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

########################### Naive Bayes  ###################################

from sklearn.naive_bayes import GaussianNB
naive = GaussianNB().fit(X_train, y_train)
y_pred_naive =naive.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
confusion_matrix(y_test,y_pred_naive)
accuracy_score(y_test,y_pred_naive)
print(classification_report(y_test,y_pred_naive))

from sklearn.metrics import roc_auc_score,roc_curve
auc = roc_auc_score(y_test,y_pred_naive);auc

y_pred_prob_naive = naive.predict_proba(X_test);y_pred_prob_naive

fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob_naive[:,1])

# ROC curve
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
print(roc_auc_score(y_test, y_pred_naive))

########################### Logistic Regression ##########################
from sklearn.linear_model import LogisticRegression
log = LogisticRegression().fit(X_train, y_train)
y_pred_log = log.predict(X_test)

confusion_matrix(y_test,y_pred_log)
accuracy_score(y_test,y_pred_log)
print(classification_report(y_test,y_pred_log))

auc_log = roc_auc_score(y_test,y_pred_log);auc_log

y_pred_prob_log = naive.predict_proba(X_test);y_pred_prob_log

fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob_log[:,1])

# ROC curve
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc_log)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Log_reg')
plt.legend()
plt.show()
print(roc_auc_score(y_test, y_pred_log))

################################## SVM ##################################
from sklearn.svm import SVC
sv = SVC(random_state = 265).fit(X_train, y_train)
y_pred_sv = sv.predict(X_test)

confusion_matrix(y_test,y_pred_sv)
accuracy_score(y_test,y_pred_sv)
print(classification_report(y_test,y_pred_sv))

auc_sv = roc_auc_score(y_test,y_pred_sv);auc_sv

y_pred_prob_sv = naive.predict_proba(X_test);y_pred_prob_sv

fpr_sv,tpr_sv,thresholds_sv = roc_curve(y_test,y_pred_prob_sv[:,1])

# ROC curve
plt.plot(fpr_sv, tpr_sv, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc_sv)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SVC')
plt.legend()
plt.show()
print(roc_auc_score(y_test, y_pred_sv))


param_grid={'C':[0.1,1,0.2,0.5],'gamma':[1,0.5,00.001]}
grid= GridSearchCV(sv,param_grid, verbose=3, n_jobs=-1)
grid.fit(X_train,y_train)
grid.best_params_

sv1 = SVC(C= 1, gamma= 0.001,random_state = 265).fit(X_train, y_train)
y_pred_sv1 = sv1.predict(X_test)

confusion_matrix(y_test,y_pred_sv1)
accuracy_score(y_test,y_pred_sv1)
print(classification_report(y_test,y_pred_sv1))

auc_sv1 = roc_auc_score(y_test,y_pred_sv1);auc_sv1

y_pred_prob_sv1 = naive.predict_proba(X_test);y_pred_prob_sv1

fpr_sv1,tpr_sv1,thresholds_sv1 = roc_curve(y_test,y_pred_prob_sv1[:,1])

# ROC curve
plt.plot(fpr_sv1, tpr_sv1, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc_sv)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SVC')
plt.legend()
plt.show()
print(roc_auc_score(y_test, y_pred_sv1))

################### Decision Tree ###################################
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 265).fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

confusion_matrix(y_test,y_pred_dt)
accuracy_score(y_test,y_pred_dt)
print(classification_report(y_test,y_pred_dt))

auc_dt = roc_auc_score(y_test,y_pred_dt);auc_dt

y_pred_prob_dt = dt.predict_proba(X_test);y_pred_prob_dt

fpr_dt,tpr_dt,thresholds_dt = roc_curve(y_test,y_pred_prob_dt[:,1])

# ROC curve
plt.plot(fpr_dt, tpr_dt, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc_sv)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Decision Tree')
plt.legend()
plt.show()
print(roc_auc_score(y_test, y_pred_dt))

leaves = [1,2,4,5,10,20,30,40,80,100]
grid_param_dt = {
    'criterion': ['gini', 'entropy'],
    'max_features' : ['auto','log2'],
    'min_samples_leaf':leaves
}
grid= GridSearchCV(dt,grid_param_dt, verbose=3, n_jobs=-1)
grid.fit(X_train,y_train)
grid.best_params_

dt1 = DecisionTreeClassifier(criterion = 'entropy', max_features = 'auto', min_samples_leaf =  1,random_state = 265).fit(X_train, y_train)
y_pred_dt1 = dt1.predict(X_test)

confusion_matrix(y_test,y_pred_dt1)
accuracy_score(y_test,y_pred_dt1)
print(classification_report(y_test,y_pred_dt1))

auc_dt1 = roc_auc_score(y_test,y_pred_dt1);auc_dt1

y_pred_prob_dt1 = dt1.predict_proba(X_test);y_pred_prob_dt1

fpr_dt1,tpr_dt1,thresholds_dt1 = roc_curve(y_test,y_pred_prob_dt1[:,1])

# ROC curve
plt.plot(fpr_dt1, tpr_dt1, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc_dt1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Decision Tree')
plt.legend()
plt.show()
print(roc_auc_score(y_test, y_pred_dt1))

############################ Random Forest #####################
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=265).fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

confusion_matrix(y_test,y_pred_rf)
accuracy_score(y_test,y_pred_rf)
print(classification_report(y_test,y_pred_rf))

auc_rf = roc_auc_score(y_test,y_pred_rf);auc_rf

y_pred_prob_rf = rf.predict_proba(X_test);y_pred_prob_rf

fpr_rf,tpr_rf,thresholds_rf = roc_curve(y_test,y_pred_prob_rf[:,1])

# ROC curve
plt.plot(fpr_rf, tpr_rf, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc_rf)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest')
plt.legend()
plt.show()
print(roc_auc_score(y_test, y_pred_rf))


leaves = [1,2,4,5,10,20,30,40,80,100]
grid_param_rf = {
    "n_estimators" : [10,20,30,40],
    'criterion': ['gini', 'entropy'],
    'max_features' : ['auto','log2'],
    'min_samples_leaf':leaves
    
}

grid_search = GridSearchCV(rf,grid_param_rf,cv=5,n_jobs =-1,verbose = 3).fit(X_train,y_train)
grid_search.best_params_

rf1 = RandomForestClassifier(criterion = 'entropy',max_features = 'log2',
                            min_samples_leaf = 1,n_estimators = 40,random_state=265)
rf1.fit(X_train,y_train)
y_pred_rf1 = rf1.predict(X_test)

confusion_matrix(y_test,y_pred_rf1)
accuracy_score(y_test,y_pred_rf1)
print(classification_report(y_test,y_pred_rf1))

auc_rf1 = roc_auc_score(y_test,y_pred_rf1);auc_rf1

y_pred_prob_rf1 = rf1.predict_proba(X_test);y_pred_prob_rf1

fpr_rf1,tpr_rf1,thresholds_rf1 = roc_curve(y_test,y_pred_prob_rf1[:,1])

# ROC curve
plt.plot(fpr_rf1, tpr_rf1, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc_rf1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest')
plt.legend()
plt.show()
print(roc_auc_score(y_test, y_pred_rf1))

############################ Ada Boost #####################
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(random_state = 265).fit(X_train,y_train)
y_pred_ada = ada.predict(X_test)

confusion_matrix(y_test,y_pred_ada)
accuracy_score(y_test,y_pred_ada)
print(classification_report(y_test,y_pred_ada))

auc_ada = roc_auc_score(y_test,y_pred_ada);auc_ada

y_pred_prob_ada = ada.predict_proba(X_test);y_pred_prob_ada

fpr_ada,tpr_ada,thresholds_ada = roc_curve(y_test,y_pred_prob_ada[:,1])

# ROC curve
plt.plot(fpr_ada, tpr_ada, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc_ada)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for AdaBoost')
plt.legend()
plt.show()
print(roc_auc_score(y_test, y_pred_ada))

grid_param_ada={'n_estimators' : [100,200,300],'learning_rate':[.001,0.01,0.1]}
grid_search_ada = GridSearchCV(ada,grid_param_ada,cv=5,n_jobs =-1,verbose = 3)
grid_search_ada.fit(X_train,y_train)
grid_search_ada.best_params_

ada1 = AdaBoostClassifier(random_state = 265,learning_rate = 0.1,n_estimators = 300).fit(X_train,y_train)
y_pred_ada1 = ada1.predict(X_test)

confusion_matrix(y_test,y_pred_ada1)
accuracy_score(y_test,y_pred_ada1)
print(classification_report(y_test,y_pred_ada1))

auc_ada1 = roc_auc_score(y_test,y_pred_ada1);auc_ada1

y_pred_prob_ada1 = ada1.predict_proba(X_test);y_pred_prob_ada1

fpr_ada1,tpr_ada1,thresholds_ada1 = roc_curve(y_test,y_pred_prob_ada1[:,1])

# ROC curve
plt.plot(fpr_ada1, tpr_ada1, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc_ada1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for AdaBoost')
plt.legend()
plt.show()
print(roc_auc_score(y_test, y_pred_ada1))

################# Gradient Boosting #####################
from sklearn.ensemble import GradientBoostingClassifier
grad = GradientBoostingClassifier(random_state = 265).fit(X_train,y_train)
y_pred_grad = grad.predict(X_test)


confusion_matrix(y_test,y_pred_grad)
accuracy_score(y_test,y_pred_grad)
print(classification_report(y_test,y_pred_grad))

auc_grad = roc_auc_score(y_test,y_pred_grad);auc_grad

y_pred_prob_grad = grad.predict_proba(X_test);y_pred_prob_grad

fpr_grad,tpr_grad,thresholds_grad = roc_curve(y_test,y_pred_prob_grad[:,1])

# ROC curve
plt.plot(fpr_grad, tpr_grad, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc_grad)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Gradient Boost')
plt.legend()
plt.show()
print(roc_auc_score(y_test, y_pred_grad))

grid_param_grad={'n_estimators' : [100,200,300],'learning_rate':[.001,0.01,0.1]}
grid_search_grad = GridSearchCV(grad,grid_param_grad,cv=5,n_jobs =-1,verbose = 3)
grid_search_grad.fit(X_train,y_train)
grid_search_grad.best_params_

grad1 = GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 300,random_state = 265).fit(X_train,y_train)
y_pred_grad1 = grad.predict(X_test)


confusion_matrix(y_test,y_pred_grad1)
accuracy_score(y_test,y_pred_grad1)
print(classification_report(y_test,y_pred_grad1))

auc_grad1 = roc_auc_score(y_test,y_pred_grad1);auc_grad1

y_pred_prob_grad1 = grad.predict_proba(X_test);y_pred_prob_grad1

fpr_grad1,tpr_grad1,thresholds_grad1 = roc_curve(y_test,y_pred_prob_grad1[:,1])

# ROC curve
plt.plot(fpr_grad1, tpr_grad1, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc_grad1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Gradient Boost')
plt.legend()
plt.show()
print(roc_auc_score(y_test, y_pred_grad1))

################################ XG Boost ###############################
import xgboost
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state = 265).fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_test)


confusion_matrix(y_test,y_pred_xgb)
accuracy_score(y_test,y_pred_xgb)
print(classification_report(y_test,y_pred_xgb))

auc_xgb = roc_auc_score(y_test,y_pred_xgb);auc_xgb

y_pred_prob_xgb = xgb.predict_proba(X_test);y_pred_prob_xgb

fpr_xgb,tpr_xgb,thresholds_xgb = roc_curve(y_test,y_pred_prob_xgb[:,1])

# ROC curve
plt.plot(fpr_xgb, tpr_xgb, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc_xgb)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for XG Boost')
plt.legend()
plt.show()
print(roc_auc_score(y_test, y_pred_xgb))


grid_param_xgb={'n_estimators' : [100,200,300],'learning_rate':[.001,0.01,0.1]}
grid_search_xgb = GridSearchCV(xgb,grid_param_xgb,cv=5,n_jobs =-1,verbose = 3)
grid_search_xgb.fit(X_train,y_train)
grid_search_xgb.best_params_


xgb1 = XGBClassifier(learning_rate = 0.1, n_estimators = 300,random_state = 265).fit(X_train,y_train)
y_pred_xgb1 = xgb1.predict(X_test)


confusion_matrix(y_test,y_pred_xgb1)
accuracy_score(y_test,y_pred_xgb1)
print(classification_report(y_test,y_pred_xgb1))

auc_xgb1 = roc_auc_score(y_test,y_pred_xgb1);auc_xgb1

y_pred_prob_xgb1 = xgb1.predict_proba(X_test);y_pred_prob_xgb1

fpr_xgb1,tpr_xgb1,thresholds_xgb1 = roc_curve(y_test,y_pred_prob_xgb1[:,1])

# ROC curve
plt.plot(fpr_xgb1, tpr_xgb1, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc_xgb1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for XG Boost')
plt.legend()
plt.show()
print(roc_auc_score(y_test, y_pred_xgb1))



##### Random Forest (gives 98% accuracy) is the best model compare to other models.