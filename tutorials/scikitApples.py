## scikit-learn decision tree example from google developers
## https://www.youtube.com/watch?v=tNa99PG8hR8&index=2&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal

from sklearn import tree

# fruit features = [weight, smoothness]
examples = [[140, 1], [130, 1], [150,0], [170,0]]

# 0 = apple, 1 = orange
labels = [0,0,1,1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(examples, labels)

print clf.predict([[160,0]])
print clf.predict([[100,1]])
print clf.predict([[100,0]])
