import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import GridSearchCV
import warnings
import pickle
warnings.filterwarnings("ignore")

#col_names = ['pregnant', 'glucose', 'BP', 'Skin Thickness', 'Insulin', 'BMI', 'pedigree', 'age', 'label']
df = pd.read_csv("diabetes.csv")
df = np.array(df)
#df = df.astype('float')

#print(df.dtypes)
#print(df.head())
X = df[1:, 1:-1]
y = df[1:, -1]
#X = X.astype('float')
#y = y.astype('float')
#print(X.shape)
#print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

logreg = LogisticRegression()
#print(cross_val_score(logreg, X, y, cv = 10, scoring = 'accuracy').mean())
logreg.fit(X_train, y_train)
#y_pred_split = logreg.predict(X_test)

pickle.dump(logreg,open('model1.pkl','wb'))
model1=pickle.load(open('model1.pkl','rb'))