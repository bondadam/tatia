import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import ast



def main():
    dataset = pd.read_csv('dataset/data.csv')
    print(dataset.head())
    X = dataset.iloc[:, 0:6].values
    y = dataset.iloc[:, 6].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #regressor = DecisionTreeClassifier(n_estimators=15, random_state=0)
    regressor = DecisionTreeClassifier()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))
    while 1:
        testValue = input("Entrez le sample généré à tester:\n")
        X_test = [ast.literal_eval(testValue)]
        result = regressor.predict(X_test)
        print("Prédiction : " + "Même auteur." if result[0] == 1 else "Auteur différent")
        restart = input("Voulez-vous tester un autre sample? (Y/N)\n")
        if restart.strip().lower() != "y":
            break

    
    
 


if __name__== "__main__":
  main()
  input("Press Enter to continue...")
