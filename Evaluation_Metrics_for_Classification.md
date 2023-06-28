# Evaluation Metrics For Classification

## **Confusion Matrix**

When creating a machine learning algorithm capable of making predictions, an important step in the process is to measure the model’s predictive power. In this lesson we will learn how to calculate some of the more common evaluation metrics for classification problems. Remember, in order to calculate these statistics, we need to split our data into a [training, validation, and test set](https://www.codecademy.com/content-items/ced99a64b810eda769bc48293550fd21) before we start fitting our model.

Let’s say we are fitting a machine learning model to try to predict whether or not an email is spam. We can pass the features of our evaluation set through the trained model and get an output list of the predictions our model makes. We then compare each of those predictions to the actual labels. There are four possible categories that each of the comparisons can fall under:

- True Positive (**TP**): The algorithm predicted spam and it was spam
- True Negative (**TN**): The algorithm predicted not spam and it was not spam
- False Positive (**FP**): The algorithm predicted spam and it was not spam
- False Negative (**FN**): The algorithm predicted not spam and it was spam

One common way to visualize these values is in a **confusion matrix**. In a confusion matrix the predicted classes are represented as columns and the actual classes are represented as rows.

[Untitled Database](https://www.notion.so/e6c75bb0003e42bbac46dab492c48a7a?pvs=21)

Let’s calculate the number of true positives, true negatives, false positives, and false negatives from the evaluation data of a classification algorithm! Then we will construct a confusion matrix.

```python
from sklearn.metrics import confusion_matrix

actual = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
predicted = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for i in range(len(predicted)):
  if predicted[i] == 1 and actual[i] == 1:
    true_positives += 1
  if predicted[i] == 0 and actual[i] == 0:
    true_negatives += 1
  if predicted[i] == 1 and actual[i] == 0:
    false_positives += 1
  if predicted[i] == 0 and actual[i] == 1:
    false_negatives += 1
    
print(true_positives)  
print(true_negatives) 
print(false_positives) 
print(false_negatives)

conf_matrix = confusion_matrix(actual, predicted)
print(conf_matrix)
```

```powershell
3
0
3
4
[[0 3]
 [4 3]]
```

## **Accuracy**

One method for determining the effectiveness of a classification algorithm is by measuring its accuracy statistic. Accuracy is calculated by finding the total number of correctly classified predictions (true positives and true negatives) and dividing by the total number of predictions.

Accuracy is defined as:

$$
\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{True Positives} + \text{True Negatives} + \text{False Positives} + \text{False Negatives}} = \frac{TP + TN}{TP + TN + FP + FN}

$$

Let’s calculate the accuracy of the classification algorithm.

```python
accuracy = (true_positives + true_negatives)/ (true_positives + true_negatives + false_positives + false_negatives)
print(accuracy)
```

## **Recall**

Accuracy can be a misleading statistic depending on our data and the problem we are trying to solve. Consider a model tasked with predicting spam in the email inboxes of top secret government employees who never use their work email addresses for online shopping or logging onto their favorite gaming apps. We can write a pretty simple and accurate classifier that always predicts False, the email is not spam. This classifier will be incredibly accurate since there are hardly ever any spam emails sent to those top secret emails, but this classifier will never be able to find the information we are actually interested in, when there is spam.

In this situation, a helpful statistic to consider is **recall**. In our example, recall measures the ratio of correct spam predictions that our classifier found to the total number of spam emails.

Recall is defined as:

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} = \frac{TP}{TP + FN}

$$

Recall is the ratio of correct positive predictions classifications made by the model to all actual positives. For the spam classifier, this would be the number of correctly labeled spam emails divided by all the emails that were actually spam in the dataset.

Our algorithm that always predicts not spam might have a very high accuracy, but it never will find any true positives, so its recall will be 0.

```python
recall = true_positives / (true_positives + false_negatives)
print(recall)
```

## **Precision**

Unfortunately, recall isn’t a perfect statistic either (spoiler alert! There is no perfect statistic). For example, we could create a spam email classifier that always returns `True`, the email is spam. This particular classifier would have low accuracy, but the recall would be 1 because it would be able to accurately find every spam email.

In this situation, a helpful statistic to understand is **precision**. In our email spam classification example, precision is the ratio of correct spam email predictions to the total number of spam predictions.

Precision is defined as:

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} = \frac{TP}{TP+FP}

$$

Precision is the ratio of correct positive classifications to all positive classifications made by the model. For the spam classifier, this would be the number of correctly labeled spam emails divided by all the emails that were correctly or incorrectly labeled spam.

The algorithm that predicts every email is spam will have a recall of 1, but it will have very low precision. It correctly predicts every spam email, but there are tons of false positives as well.

```python
precision = true_positives / (true_positives + false_positives)
print(precision)
```

## **F1-Score**

It is often useful to consider both the precision and recall when attempting to describe the effectiveness of a model. The **F1-score** combines both precision and recall into a single statistic, by determining their harmonic mean. The harmonic mean is a method of averaging.

F1-score is defined as:

$$
\text{F1-score} = \frac{2\times \text{precision} \times \text{recall}}{\text{precision} + \text{recall}}

$$

We use the harmonic mean rather than the traditional arithmetic mean because we want the F1-score to have a low value when either precision or recall is 0.

For example, consider a classifier where `recall = 1` and `precision = 0.02`. Despite our classifier having an extremely high recall score, there is most likely a problem with this model since the precision is so low. Ideally the F1-score would reflect that.

If we took the arithmetic mean of precision and recall, we get:

$$
\frac{1 + 0.02}{2} = 0.51

$$

That performance statistic is misleadingly high for a classifier that has such dismal precision. If we instead calculate the harmonic mean, we get:

$$
\frac{2 \times 1 \times 0.02}{1 + 0.02} = 0.039

$$

That is a much better descriptor of the classifier’s effectiveness!

```python
recall = true_positives / (true_positives + false_negatives)

precision = true_positives / (true_positives + false_positives)

f_1 = (2 * recall * precision) / (recall + precision)
print(f_1)
```

## **Review**

There is no perfect metric. The decision to use accuracy, precision, recall, F1-score, or another metric not covered in this lesson ultimately comes down to the specific context of the classification problem.

Take the email spam problem. We probably want a model that is high precision and do not mind as much if it has a low recall score. This is because we want to make sure the algorithm does not incorrectly send an important email message to the spam folder, while it is not as detrimental to have a few spam emails end up in our inbox.

As long as you have an understanding of what question you’re trying to answer, you should be able to determine which statistic is most relevant to you.

The Python library `scikit-learn` has some functions that will calculate these statistics for you!

You have now learned many different ways to analyze the predictive power of your classification algorithm. Here are some of the key takeaways:

- Classifying a single point can result in a true positive (`actual = 1`, `predicted = 1`), a true negative (`actual = 0`, `predicted = 0`), a false positive (`actual = 0`, `predicted = 1`), or a false negative (`actual = 1`, `predicted = 0`). These values are often summarized in a confusion matrix.
- Accuracy measures how many classifications your algorithm got correct out of every classification it made.
- Recall is the ratio of correct positive predictions classifications made by the model to all actual positives.
- Precision is the ratio of correct positive classifications to all positive classifications made by the model.
- F1-score is a combination of precision and recall.
- F1-score will be low if either precision or recall is low.

```python
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

actual = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
predicted = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

print(accuracy_score(actual, predicted))
print(recall_score(actual, predicted))
print(precision_score(actual, predicted))
print(f1_score(actual, predicted))
```