# Random Forests

## **Random Forest**

We’ve seen that decision trees can be powerful supervised machine learning models. However, they’re not without their weaknesses — decision trees are often prone to overfitting.

We’ve discussed some strategies to minimize this problem, like pruning, but sometimes that isn’t enough. We need to find another way to generalize our trees. This is where the concept of a *random forest* comes in handy.

A *random forest* is an ensemble machine learning technique — a random forest contains many decision trees that all work together to classify new points. When a random forest is asked to classify a new point, the random forest gives that point to each of the decision trees. Each of those trees reports their classification and the random forest returns the most popular classification. It’s like every tree gets a vote, and the most popular classification wins.

Some of the trees in the random forest may be overfit, but by making the prediction based on a large number of trees, overfitting will have less of an impact.

 

## **Bagging**

You might be wondering how the trees in the random forest get created. After all, right now, our algorithm for creating a decision tree is deterministic — given a training set, the same tree will be made every time.

Random forests create different trees using a process known as *bagging*. Every time a decision tree is made, it is created using a different subset of the points in the training set. For example, if our training set had `1000` rows in it, we could make a decision tree by picking `100` of those rows at random to build the tree. This way, every tree is different, but all trees will still be created from a portion of the training data.

One thing to note is that when we’re randomly selecting these `100` rows, we’re doing so *with replacement*. Picture putting all `100` rows in a bag and reaching in and grabbing one row at random. After writing down what row we picked, we put that row back in our bag.

This means that when we’re picking our `100` random rows, we could pick the same row more than once. In fact, it’s very unlikely, but all `100` randomly picked rows could all be the same row!

Because we’re picking these rows with replacement, there’s no need to shrink our bagged training set from `1000` rows to `100`. We can pick `1000` rows at random, and because we can get the same row more than once, we’ll still end up with a unique data set.

## **Bagging Features**

We’re now making trees based on different random subsets of our initial dataset. But we can continue to add variety to the ways our trees are created by changing the features that we use.

Recall that for our car data set, the original features were the following:

- The price of the car
- The cost of maintenance
- The number of doors
- The number of people the car can hold
- The size of the trunk
- The safety rating

Right now when we create a decision tree, we look at every one of those features and choose to split the data based on the feature that produces the most information gain. We could change how the tree is created by only allowing a subset of those features to be considered at each split.

For example, when finding which feature to split the data on the first time, we might randomly choose to only consider the price of the car, the number of doors, and the safety rating.

After splitting the data on the best feature from that subset, we’ll likely want to split again. For this next split, we’ll randomly select three features again to consider. This time those features might be the cost of maintenance, the number of doors, and the size of the trunk. We’ll continue this process until the tree is complete.

One question to consider is how to choose the number of features to randomly select. Why did we choose `3` in this example? A good rule of thumb is to randomly select the square root of the total number of features. Our car dataset doesn’t have a lot of features, so in this example, it’s difficult to follow this rule. But if we had a dataset with `25` features, we’d want to randomly select `5` features to consider at every split point.

```python
from tree import car_data, car_labels, split, information_gain
import random
import numpy as np
np.random.seed(1)
random.seed(4)

def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    #Create features here
    features = np.random.choice(len(dataset[0]), 3, replace=False)
    for feature in features:
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_gain, best_feature
  
indices = [random.randint(0, 999) for i in range(1000)]

data_subset = [car_data[index] for index in indices]
labels_subset = [car_labels[index] for index in indices]
print(find_best_split(data_subset, labels_subset))
```

## **Classify**

Now that we can make different decision trees, it’s time to plant a whole forest! Let’s say we make different `8` trees using bagging and feature bagging. We can now take a new unlabeled point, give that point to each tree in the forest, and count the number of times different labels are predicted.

The trees give us their votes and the label that is predicted most often will be our final classification! For example, if we gave our random forest of 8 trees a new data point, we might get the following results:

```
["vgood", "vgood", "good", "vgood", "acc", "vgood", "good", "vgood"]

```

Since the most commonly predicted classification was `"vgood"`, this would be the random forest’s final classification.

```python
from tree import build_tree, print_tree, car_data, car_labels, classify
import random
random.seed(4)

# The features are the price of the car, the cost of maintenance, the number of doors, the number of people the car can hold, the size of the trunk, and the safety rating
unlabeled_point = ['high', 'vhigh', '3', 'more', 'med', 'med']

predictions = []
for i in range(20):
  indices = [random.randint(0, 999) for i in range(1000)]
  data_subset = [car_data[index] for index in indices]
  labels_subset = [car_labels[index] for index in indices]
  subset_tree = build_tree(data_subset, labels_subset)

  predictions.append(classify( unlabeled_point, subset_tree))

print(predictions)
final_prediction = max(predictions, key=predictions.count)
print(final_prediction)
```

## **Test Set**

We’re now able to create a random forest, but how accurate is it compared to a single decision tree? To answer this question we’ve split our data into a training set and test set. By building our models using the training set and testing on every data point in the test set, we can calculate the accuracy of both a single decision tree and a random forest.

We’ve given you code that calculates the accuracy of a single tree. This tree was made without using any of the bagging techniques we just learned. We created the tree by using every row from the training set once and considered every feature when splitting the data rather than a random subset.

```python
from tree import training_data, training_labels, testing_data, testing_labels, make_random_forest, make_single_tree, classify
import numpy as np
import random
np.random.seed(1)
random.seed(1)
from collections import Counter

tree = make_single_tree(training_data, training_labels)
single_tree_correct = 0

forest = make_random_forest(40, training_data, training_labels)
forest_correct = 0

for i in range(len(testing_data)):
  prediction = classify(testing_data[i], tree)
  if prediction == testing_labels[i]:
    single_tree_correct += 1
  predictions = []
  for forest_tree in forest:
    predictions.append(classify(testing_data[i], forest_tree))
  forest_prediction = max(predictions,key=predictions.count)
  if forest_prediction == testing_labels[i]:
    forest_correct += 1
    
print(single_tree_correct/len(testing_data))
print(forest_correct/len(testing_data))
```

## **Random Forest in Scikit-learn**

You now have the ability to make a random forest using your own decision trees. However, `scikit-learn` has a `RandomForestClassifier` class that will do all of this work for you! `RandomForestClassifier` is in the `sklearn.ensemble` module.

`RandomForestClassifier` works almost identically to `DecisionTreeClassifier` — the `.fit()`, `.predict()`, and `.score()` methods work in the exact same way.

When creating a `RandomForestClassifier`, you can choose how many trees to include in the random forest by using the `n_estimators` parameter like this:

```python
classifier = RandomForestClassifier(n_estimators = 100)
```

We now have a very powerful machine learning model that is fairly resistant to overfitting!

```python
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 2000, random_state = 0)
classifier.fit(training_points, training_labels)
print(classifier.score(testing_points, testing_labels))
```

**Review**

Nice work! Here are some of the major takeaways about random forests:

- A random forest is an ensemble machine learning model. It makes a classification by aggregating the classifications of many decision trees.
- Random forests are used to avoid overfitting. By aggregating the classification of multiple trees, having overfitted trees in a random forest is less impactful.
- Every decision tree in a random forest is created by using a different subset of data points from the training set. Those data points are chosen at random *with replacement*, which means a single data point can be chosen more than once. This process is known as *bagging*.
- When creating a tree in a random forest, a randomly selected subset of features are considered as candidates for the best splitting feature. If your dataset has `n` features, it is common practice to randomly select the square root of `n` features.