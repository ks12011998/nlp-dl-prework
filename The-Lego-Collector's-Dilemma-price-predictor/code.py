# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
df = pd.read_csv(path)
df.head()

X = pd.DataFrame(df,columns=['ages','num_reviews','piece_count','play_star_rating','review_difficulty','star_rating','theme_name','val_star_rating','country'])

y = df['list_price']

X_train,y_train,X_test,y_test = train_test_split(X,y,random_state=6,test_size=0.3)


# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here        
cols = X_train.columns

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20,20))

for i in range(0,3):
    for j in range(0,3): 
            col = cols[i*3 + j]
            axes[i,j].set_title(col)
            axes[i,j].scatter(X_train[col],y_train)
            axes[i,j].set_xlabel(col)
            axes[i,j].set_ylabel('list_price')
        

# code ends here
plt.show()


# --------------
# Code starts here

corr =  X_train.corr(method='pearson')
print(corr)
# Code ends here

del X_train['play_star_rating']
del X_train['val_star_rating']

del X_test['play_star_rating']
del X_test['val_star_rating']



# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here
regressor =  LinearRegression()

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

mse  = mean_squared_error(y_test,y_pred)
print(mse)

r2 = r2_score(y_test,y_pred)
print(r2)

# Code ends here


# --------------
# Code starts here


residual = y_test  -  y_pred

hist = np.histogram(residual)
print(hist)

# Code ends here


