import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import ast
import graphviz
from sklearn.tree.export import export_text
import os


def main():
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    dataset = pd.read_csv('dataset/data.csv')
    print(dataset.head())
    X = dataset.iloc[:, 0:6].values
    y = dataset.iloc[:, 6].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    decisiontree = DecisionTreeClassifier()
    newDecisiontree = decisiontree.fit(X_train, y_train)
    dot_data = tree.export_graphviz(newDecisiontree, out_file=None, feature_names=dataset.columns[:-1].values.tolist(),class_names=["Auteur différent", "Même auteur"],filled=True, rounded=True,) 
    graph = graphviz.Source(dot_data) 
    graph.render("tree") 
    print(dataset.columns[:-1].values)
    r = export_text(newDecisiontree, feature_names=dataset.columns[:-1].values.tolist())
    print(r)
    y_pred = decisiontree.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))
    while 1:
        testValue = input("Entrez le sample généré à tester:\n")
        X_test = [ast.literal_eval(testValue)]
        result = decisiontree.predict(X_test)
        print("Prédiction : Même auteur." if result[0] == 1 else "Prédiction : Auteur différent")
        restart = input("Voulez-vous tester un autre sample? (Y/N)\n")
        if restart.strip().lower() != "y":
            break

    
    
 


if __name__== "__main__":
  main()
  input("Press Enter to continue...")
