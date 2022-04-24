import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

# Import Flags CSV
flags = pd.read_csv("flags.csv", header = 0)

# Create labels - landmass
labels = flags["Landmass"]
# print(labels)

# Data from colours, changed at 13
data = flags[["Red", "Green", "Blue", "Gold",
 "White", "Black", "Orange",
 "Circles",
"Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"]]
# print(data)

# Split data into train and test 
x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state = 1)

# Test & Prune Tree
scores = []
for i in range(20):
  test_tree = DecisionTreeClassifier(random_state = 1, max_depth = i+1)
  test_tree.fit(x_train, y_train)
  score = test_tree.score(x_test, y_test)
  scores.append(score)

# Plot Results
# plt.plot(range(1, 21), scores)
# plt.show()

# Create Decision Tree with the determined max depth
mytree = DecisionTreeClassifier(random_state = 1, max_depth = 3)
mytree.fit(x_train, y_train)

# Score the tree
score = mytree.score(x_test, y_test)
# print(score)
print(mytree)

#  Print out tree
text_tree = tree.export_text(mytree)
print(text_tree)
