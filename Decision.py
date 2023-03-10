"""HOW TO TRAIN A MODEL TO MAKE ACCURATE DECISION"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_text_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
# 21 and 22 is assigned to the age, and 0, 1 is assigned to gender
predictions = model.predict(X_test) #( [21, 1], [22, 0] ) 

score = accuracy_score(y_test, predictions)
score
