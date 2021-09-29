################################################################################
#               Importations des bibliothèques:
################################################################################
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


################################################################################
#               Constante :
################################################################################
seed = 1


################################################################################
#               Chargement des données :
################################################################################
df = pd.read_csv('Ressources/Titanic.csv', sep=';', index_col='PassengerId')


################################################################################
#               Nettoyage des données :
################################################################################
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
df = df.dropna(subset=['Age'])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})


################################################################################
#               Split pour entrainement des modeles :
################################################################################
X = df.drop('Survived', axis=1)
Y = df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=seed)


################################################################################
#               Stochastic Gradient Descent (SGD) :
################################################################################
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

# acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd = round(accuracy_score(Y_test, Y_pred) * 100, 2)


################################################################################
#               Random Forest :
################################################################################
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

# acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest = round(accuracy_score(Y_test, Y_prediction) * 100, 2)


################################################################################
#               Logistic Regression :
################################################################################
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

# acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log = round(accuracy_score(Y_test, Y_pred) * 100, 2)


################################################################################
#               K Nearest Neighbor (KNN) :
################################################################################
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

# acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn = round(accuracy_score(Y_test, Y_pred) * 100, 2)


################################################################################
#               Gaussian Naive Bayes :
################################################################################
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

# acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian = round(accuracy_score(Y_test, Y_pred) * 100, 2)


################################################################################
#               Perceptron :
################################################################################
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

# acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron = round(accuracy_score(Y_test, Y_pred) * 100, 2)


################################################################################
#               Linear Support Vector Machine :
################################################################################
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

# acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc = round(accuracy_score(Y_test, Y_pred) * 100, 2)


################################################################################
#               Arbre de décision :
################################################################################
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

# acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree = round(accuracy_score(Y_test, Y_pred) * 100, 2)


################################################################################
#               Affichage des scores de chaques modèles :
################################################################################
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent',
              'Arbre de décision'],
    'Score': [acc_linear_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
print(result_df)
