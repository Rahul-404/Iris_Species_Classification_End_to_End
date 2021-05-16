import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# reading data
df = pd.read_csv('iris.csv')

def convert_to_int(word):
    word_dict = {'setosa' : 0,
                    'versicolor' : 1,
                    'virginica' : 2
    }
    return word_dict[word]

# encoding target
df['species'] = df['species'].apply(lambda x : convert_to_int(x))

FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
TARGET = 'species'

# X and y separation
X = df[FEATURES].values
y = df[TARGET].values

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify = y)

# complete pipeline for training
# model = make_pipeline(StandardScaler(), GaussianNB(priors=None))
# _ = model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print('Model accuracy: %r'% acc)

# saving model for production
model = make_pipeline(StandardScaler(), GaussianNB(priors=None))
_ = model.fit(X, y)

file_obj = open(b"model.pkl", "wb")
pickle.dump(model, file_obj)

print('Model created for production')