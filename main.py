import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score





def main():
    dataset = pd.read_csv('dataset/data.csv')
    print(dataset.head())
    X = dataset.iloc[:, 0:6].values
    y = dataset.iloc[:, 6].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = RandomForestClassifier(n_estimators=15, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

    
    #for i in range(100):
    #    xnew = [dataset.iloc[i, 0:6].values]
    #    ynew = regressor.predict(xnew)
    #    #print("la javanaise vs histoire provernce : " + ynew)
    #    print("X=%s, Predicted=%s" % (xnew[0], ynew[0]))

    xnew = [[0.9991592738535356,0.9448425671335888,0.9970761020439538,0.8479555244818331,0.95674518201284795,0.9385236263064318]]
    ynew = regressor.predict(xnew)
    #print("la javanaise vs histoire provernce : " + ynew)
    print("X=%s, Predicted=%s" % (xnew[0], ynew[0]))


if __name__== "__main__":
  main()
  input("Press Enter to continue...")
