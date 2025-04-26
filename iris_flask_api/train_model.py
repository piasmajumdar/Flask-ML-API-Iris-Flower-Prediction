from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

iris = load_iris()
X, y = iris.data, iris.target

clf = DecisionTreeClassifier()
clf.fit(X, y)

joblib.dump(clf, 'iris_tree.joblib')
print("Model trained and saved to iris_tree.joblib")
