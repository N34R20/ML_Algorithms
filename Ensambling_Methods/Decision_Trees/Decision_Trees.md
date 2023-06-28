# Decision Trees

## **Decision Trees**

Decision trees are machine learning models that try to find patterns in the features of data points. Take a look at the tree on this page. This tree tries to predict whether a student will get an A on their next test.

By asking questions like “What is the student’s average grade in the class” the decision tree tries to get a better understanding of their chances on the next test.

In order to make a classification, this classifier needs a data point with four features:

- The student’s average grade in the class.
- The number of hours the student plans on studying for the test.
- The number of hours the student plans on sleeping the night before the test.
- Whether or not the student plans on cheating.

For example, let’s say that somebody has a “B” average in the class, studied for more than 3 hours, slept less than 5 hours before the test, and doesn’t plan to cheat. If we start at the top of the tree and take the correct path based on that data, we’ll arrive at a *leaf node* that predicts the person will *not* get an A on the next test.

## **Making Decision Trees**

If we’re given this magic tree, it seems relatively easy to make classifications. But how do these trees get created in the first place? Decision trees are supervised machine learning models, which means that they’re created from a training set of labeled data. Creating the tree is where the *learning* in machine learning happens.

Take a look at the gif on this page. We begin with every point in the training set at the top of the tree. These training points have labels — the red points represent students that didn’t get an A on a test and the green points represent students that did get an A on a test .

We then decide to split the data into smaller groups based on a feature. For example, that feature could be something like their average grade in the class. Students with an A average would go into one set, students with a B average would go into another subset, and so on.

Once we have these subsets, we repeat the process — we split the data in each subset again on a different feature.

Eventually, we reach a point where we decide to stop splitting the data into smaller groups. We’ve reached a leaf of the tree. We can now count up the labels of the data in that leaf. If an unlabeled point reaches that leaf, it will be classified as the majority label.

We can now make a tree, but how did we know which features to split the data set with? After all, if we started by splitting the data based on the number of hours they slept the night before the test, we’d end up with a very different tree that would produce very different results. How do we know which tree is best? We’ll tackle this question soon!

## **Gini Impurity**

Consider the two trees below. Which tree would be more useful as a model that tries to predict whether someone would get an A in a class?

![https://content.codecademy.com/programs/data-science-path/decision-trees/comparison_1.svg](https://content.codecademy.com/programs/data-science-path/decision-trees/comparison_1.svg)

![https://content.codecademy.com/programs/data-science-path/decision-trees/comparison_2.svg](https://content.codecademy.com/programs/data-science-path/decision-trees/comparison_2.svg)

Let’s say you use the top tree. You’ll end up at a leaf node where the label is up for debate. The training data has labels from both classes! If you use the bottom tree, you’ll end up at a leaf where there’s only one type of label. There’s no debate at all! We’d be much more confident about our classification if we used the bottom tree.

This idea can be quantified by calculating the *Gini impurity* of a set of data points. To find the Gini impurity, start at `1` and subtract the squared percentage of each label in the set. For example, if a data set had three items of class `A` and one item of class `B`, the Gini impurity of the set would be

$$
1 - \bigg(\frac{3}{4}\bigg)^2 - \bigg(\frac{1}{4}\bigg) = 0.375

$$

If a data set has only one class, you’d end up with a Gini impurity of `0`. The lower the impurity, the better the decision tree!

```sql
from collections import Counter

labels = ["unacc", "unacc", "acc", "acc", "good", "good"]
#labels = ["unacc","unacc","unacc", "good", "vgood", "vgood"]
#labels = ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc"]

impurity = 1

label_counts = Counter(labels)
print(label_counts)

for label in label_counts:
  probability_of_label = label_counts[label]/len(labels)
  impurity -= probability_of_label**2

print(impurity)
```

## **Information Gain**

We know that we want to end up with leaves with a low Gini Impurity, but we still need to figure out which features to split on in order to achieve this. For example, is it better if we split our dataset of students based on how much sleep they got or how much time they spent studying?

To answer this question, we can calculate the *information gain* of splitting the data on a certain feature. Information gain measures difference in the impurity of the data before and after the split. For example, let’s say you had a dataset with an impurity of `0.5`. After splitting the data based on a feature, you end up with three groups with impurities `0`, `0.375`, and `0`. The information gain of splitting the data in that way is `0.5 - 0 - 0.375 - 0 = 0.125`.

![https://content.codecademy.com/programs/data-science-path/decision-trees/info.svg](https://content.codecademy.com/programs/data-science-path/decision-trees/info.svg)

Not bad! By splitting the data in that way, we’ve gained some information about how the data is structured — the datasets after the split are purer than they were before the split. The higher the information gain the better — if information gain is `0`, then splitting the data on that feature was useless! Unfortunately, right now it’s possible for information gain to be negative. In the next exercise, we’ll calculate *weighted* information gain to fix that problem.

```sql
from collections import Counter

unsplit_labels = ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc", "good", "good", "good", "good", "vgood", "vgood", "vgood"]

split_labels_1 = [
  ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc", "good", "good", "vgood"], 
  [ "good", "good"], 
  ["vgood", "vgood"]
]

split_labels_2 = [
  ["unacc", "unacc", "unacc", "unacc","unacc", "unacc", "good", "good", "good", "good"], 
  ["vgood", "vgood", "vgood"]
]

def gini(dataset):
  impurity = 1
  label_counts = Counter(dataset)
  for label in label_counts:
    prob_of_label = label_counts[label] / len(dataset)
    impurity -= prob_of_label ** 2
  return impurity

info_gain = gini(unsplit_labels)
print(info_gain)

for subset in split_labels_1:
  info_gain -= gini(subset)
print(info_gain)
```

## **Weighted Information Gain**

We’re not quite done calculating the information gain of a set of objects. The sizes of the subset that get created after the split are important too! For example, the image below shows two sets with the same impurity. Which set would you rather have in your decision tree?

![https://content.codecademy.com/programs/data-science-path/decision-trees/impurity-0.svg](https://content.codecademy.com/programs/data-science-path/decision-trees/impurity-0.svg)

Both of these sets are perfectly pure, but the purity of the second set is much more meaningful. Because there are so many items in the second set, we can be confident that whatever we did to produce this set wasn’t an accident.

It might be helpful to think about the inverse as well. Consider these two sets with the same impurity:

![https://content.codecademy.com/programs/data-science-path/decision-trees/impurity-5.svg](https://content.codecademy.com/programs/data-science-path/decision-trees/impurity-5.svg)

Both of these sets are completely impure. However, that impurity is much more meaningful in the set with more instances. We know that we are going to have to do a lot more work in order to completely separate the two classes. Meanwhile, the impurity of the set with two items isn’t as important. We know that we’ll only need to split the set one more time in order to make two pure sets.

Let’s modify the formula for information gain to reflect the fact that the size of the set is relevant. Instead of simply subtracting the impurity of each set, we’ll subtract the *weighted* impurity of each of the split sets. If the data before the split contained `20` items and one of the resulting splits contained `2` items, then the weighted impurity of that subset would be `2/20 * impurity`. We’re lowering the importance of the impurity of sets with few elements.

![https://content.codecademy.com/programs/data-science-path/decision-trees/weighted_info.svg](https://content.codecademy.com/programs/data-science-path/decision-trees/weighted_info.svg)

Now that we can calculate the information gain using weighted impurity, let’s do that for every possible feature. If we do this, we can find the best feature to split the data on.

## **Recursive Tree Building**

Now that we can find the best feature to split the dataset, we can repeat this process again and again to create the full tree. This is a recursive algorithm! We start with every data point from the training set, find the best feature to split the data, split the data based on that feature, and then recursively repeat the process again on each subset that was created from the split.

We’ll stop the recursion when we can no longer find a feature that results in any information gain. In other words, we want to create a leaf of the tree when we can’t find a way to split the data that makes purer subsets.

The leaf should keep track of the classes of the data points from the training set that ended up in the leaf. In our implementation, we’ll use a `Counter` object to keep track of the counts of labels.

We’ll use these counts to make predictions about new data that we give the tree.

## **Classifying New Data**

We can finally use our tree as a classifier! Given a new data point, we start at the top of the tree and follow the path of the tree until we hit a leaf. Once we get to a leaf, we’ll use the classes of the points from the training set to make a classification.

We’ve slightly changed the way our `build_tree()` function works. Instead of returning a list of branches or a `Counter` object, the `build_tree()` function now returns a `Leaf` object or an `Internal_Node` object. We’ll explain how to use these objects in the instructions!

## **Decision Trees in scikit-learn**

Nice work! You’ve written a decision tree from scratch that is able to classify new points. Let’s take a look at how the Python library `scikit-learn` implements decision trees.

The `sklearn.tree` module contains the `DecisionTreeClassifier` class. To create a `DecisionTreeClassifier` object, call the constructor:

```python
classifier = DecisionTreeClassifier()

```

Next, we want to create the tree based on our training data. To do this, we’ll use the `.fit()` method.

`.fit()` takes a list of data points followed by a list of the labels associated with that data. Note that when we built our tree from scratch, our data points contained strings like `"vhigh"` or `"5more"`. When creating the tree using `scikit-learn`, it’s a good idea to map those strings to numbers. For example, for the first feature representing the price of the car, `"low"` would map to `1`, `"med"` would map to `2`, and so on.

```python
classifier.fit(training_data, training_labels)

```

Finally, once we’ve made our tree, we can use it to classify new data points. The `.predict()` method takes an array of data points and will return an array of classifications for those data points.

```python
predictions = classifier.predict(test_data)

```

If you’ve split your data into a test set, you can find the accuracy of the model by calling the `.score()` method using the test data and the test labels as parameters.

```python
print(classifier.score(test_data, test_labels))

```

`.score()` returns the percentage of data points from the test set that it classified correctly.

## **Decision Tree Limitations**

Now that we have an understanding of how decision trees are created and used, let’s talk about some of their limitations.

One problem with the way we’re currently making our decision trees is that our trees aren’t always *globablly optimal*. This means that there might be a better tree out there somewhere that produces better results. But wait, why did we go through all that work of finding information gain if it’s not producing the best possible tree?

Our current strategy of creating trees is *greedy*. We assume that the best way to create a tree is to find the feature that will result in the largest information gain *right now* and split on that feature. We never consider the ramifications of that split further down the tree. It’s possible that if we split on a suboptimal feature right now, we would find even better splits later on. Unfortunately, finding a globally optimal tree is an extremely difficult task, and finding a tree using our greedy approach is a reasonable substitute.

Another problem with our trees is that they potentially *overfit* the data. This means that the structure of the tree is too dependent on the training data and doesn’t accurately represent the way the data in the real world looks like. In general, larger trees tend to overfit the data more. As the tree gets bigger, it becomes more tuned to the training data and it loses a more generalized understanding of the real world data.

One way to solve this problem is to *prune* the tree. The goal of pruning is to shrink the size of the tree. There are a few different pruning strategies, and we won’t go into the details of them here. `scikit-learn` currently doesn’t prune the tree by default, however we can dig into the code a bit to prune it ourselves.

```python
classifier = DecisionTreeClassifier(random_state = 0, max_depth=11)
print(classifier.tree_.max_depth)
```

## **Review**

Great work! In this lesson, you learned how to create decision trees and use them to make classifications. Here are some of the major takeaways:

- Good decision trees have pure leaves. A leaf is pure if all of the data points in that class have the same label.
- Decision trees are created using a greedy algorithm that prioritizes finding the feature that results in the largest information gain when splitting the data using that feature.
- Creating an optimal decision tree is difficult. The greedy algorithm doesn’t always find the globally optimal tree.
- Decision trees often suffer from overfitting. Making the tree small by pruning helps to generalize the tree so it is more accurate on data in the real world.