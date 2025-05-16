import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"\nAccuracy on test data: {accuracy * 100:.2f}%")
new_sample = [[5.1, 3.5, 1.4, 0.2]] 
predicted_class = classifier.predict(new_sample)
print(f"\nNew sample classified as: {iris.target_names[predicted_class][0]}")
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
tree.plot_tree(classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()