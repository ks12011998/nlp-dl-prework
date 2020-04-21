# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df = pd.read_csv(path)
df.head()

X = df.iloc[:,0:7]
y = df['insuranceclaim']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=6,test_size=0.2)

print(X_train)

# Code ends here


# --------------
import matplotlib.pyplot as plt


# Code starts here
boxplot = X_train.boxplot(column=['bmi'])

q_value = X_train['bmi'].quantile(.95)

print(y_train.value_counts())
# Code ends here


# --------------
# Code starts here
import matplotlib.pyplot as plt
import seaborn as sns
#pearson correlation
relation = X_train.corr(method='pearson')
print(relation)

sns.pairplot(data=X_train[["age","sex","bmi","children","smoker","region","charges"]], dropna=True)
# Code ends here


# --------------
import seaborn as sns
import matplotlib.pyplot as plt

# Code starts here
cols = ['children','sex','region','smoker']

fig,axes = plt.subplots(nrows=2,ncols=2)
print(fig)

for i in range(2):
    for j in range(2):
        col = cols[i*2+j]
        sns.countplot(x=X_train[col], hue=y_train, ax = axes[i,j])

# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here
lr = LogisticRegression(random_state=9)

grid = GridSearchCV(estimator=lr,param_grid=parameters,n_jobs=-1)
grid.fit(X_train,y_train)

y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

# Code ends here


# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Code starts here
score = roc_auc_score(y_test,y_pred)
y_pred_proba_total = grid.predict_proba(X_test)

y_pred_proba = []
for i in range(len(y_pred_proba_total)):
    y_pred_proba.append(y_pred_proba_total[i][1])

print(y_pred_proba)
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred, pos_label=2)
roc_auc = roc_auc_score(y_test,y_pred_proba)

plt.plot(fpr,tpr,label="Logistic model, auc="+str(roc_auc))

# Code ends here


