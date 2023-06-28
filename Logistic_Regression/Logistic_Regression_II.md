# Logistic Regression II

## **Assumptions of Logistic Regression I**

We’re now ready to delve deeper into Logistic Regression! In this lesson, we will cover the different assumptions that go into logistic regression, model hyperparameters, how to evaluate a classifier, ROC curves and what do when there’s a class imbalance in the classification problem we’re working with.

For this lesson, we will be using the [Wisconsin Breast Cancer Data Set](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data) (Diagnostic) to predict whether a tumor is benign or malignant based on characteristics of the cells, such as radius, texture, smoothness, etc. Like a lot of real-world data sets, the distribution of outcomes is uneven (benign diagnoses are more common than malignant) and there is a bias in terms of importance in the outcomes (classifying all malignant cases correctly is of the utmost importance).

We’re going to begin with the primary assumptions about the data that need to be checked before implementing a logistic regression model.

### **1. Target variable is binary**

One of the most basic assumptions of logistic regression is that the outcome variable needs to be binary (or in the case of multinomial LR, discrete).

### **2. Independent observations**

While often overlooked, checking for independent observations in a data set is important for the theory behind LR. This can be violated if, in this case, patients are biopsied multiple times (repeated sampling of the same individual).

### **3. Large enough sample size**

Since logistic regression is fit using maximum likelihood estimation instead of least squares minimization, there needs to be a large enough sample to get convergence. Now, what does “large enough” mean – that is often up to interpretation or the individual. But often a rule of thumb is that there should be at least 10 samples per features per class.

### **4. No influential outliers**

Logistic regression is sensitive to outliers, so it is also needed to check if there are any influential outliers for model building. Outliers are a broad topic with a lot of different definitions – z-scores, scaler of the interquartile range, Cook’s distance/influence/leverage, etc – so there are many ways to identify them. But here, we will use visual tools to rule-out obvious outliers.

## **Assumptions of Logistic Regression II**

### **1. Features linearly related to log odds**

Similar to linear regression, the underlying assumption of logistic regression is that the features are linearly related to the logit of the outcome. To test this visually, we can use seaborn’s regplot, with the parameter ‘logistic= True’ and the x value our feature of interest. If this condition is met, the model fit will resemble a sigmoidal curve (as in the case when ‘x=radius_mean’). We’ve created written code here to a second plot here using the feature ‘fractal_dimension_mean’. Press Run on the code in the workspace. How do the curves compare?

### **2. Multicollinearity**

Like in linear regression, one of the assumptions is that there is not multicolinearity in the data. There are many ways to look at this, the most common are a correlation of features and variance inflation factor (VIF). With a correlation plot, features that are highly correlated can be dropped from the model to reduce duplication.

We’re going to look at the “mean” features and which are highly correlated which each other using a heatmap. Uncomment the relevant lines of code and press Run to see the heatmap. There are two features that are highly positively correlated with radius. Can you spot them?*

- The heatmap shows radius and perimeter and area are all highly positively correlated (think formula for area of a circle!).

```python
#Compare the curves
sns.regplot(x= 'radius_mean', y= 'diagnosis', data= df, logistic= True,)
plt.show()
plt.close()

sns.regplot(x= 'fractal_dimension_mean', y= 'diagnosis', data= df, logistic= True)
plt.show()
plt.close()

# Plot a heatmap of the feature. Identify the two features that are highly correlated with radius_mean.
plt.figure(figsize = (10,7))
sns.heatmap(x.corr(), annot=True)
plt.show()
```

## **Logistic Regression Parameters in `scikit-learn`**

### **Model Training and Hyperparameters**

Now that we have checked the assumptions of Logistic Regresion, we can eliminate the appropriate features and train and predict a model using `scikit-learn`. To get the same results, some of the *hyperparameters* in the models will have to be specified from their default values. Hyperparameters are model settings that can be preset before the model implementation step and tuned later to improve model performance. They differ from parameters of a model (in the case of Logistic Regression, the feature coefficients) in that they are not the result of model implementation.

### **Evaluation Metrics**

Despite the name, logistic regression is a classifier, so any evaluation metrics for classification tasks will apply. The simplest metric is accuracy – how many correct predictions did we make out of the total? However, when classes are imbalanced, this can be a misleading metric for model performance. Similarly, if we care more about accurately predicting a certain class, other metrics may be more appropriate to use, such as precision, recall, or F1-score may be better to evaluate performance. All of these metrics are available in `scikit-learn`. Check out the [Evaluation Metrics](https://www.codecademy.com/paths/machine-learning-engineer/tracks/mle-machine-learning-fundamentals/modules/mlecp-supervised-learning-i-regressors-classifiers-and-trees/lessons/mlfun-evaluation-metrics-classification/exercises/confusion-matrix) lesson if you’d like to brush up on the same.

### **Which metrics matter most?**

For our breast cancer dataset, predicting ALL malignant cases as malignant is of the utmost importance – and even if there are some false-positives (benign cases that are marked as malignant), these likely will be discovered by follow-up tests. Whereas missing a malignant case (classifying is as benign) could have deadly consequences. Thus, we want to minimize false-negatives, which maximizes the ratio true-positives/(true-positives+false-negatives), which is recall (or sensitivity or true positive rate).

```python
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns

#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

#https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
df = pd.read_csv('breast_cancer_data.csv')
#encode malignant as 1, benign as 0
df['diagnosis'] = df['diagnosis'].replace({'M':1,'B':0})
predictor_var = ['radius_mean', 'texture_mean', 
                  'compactness_mean',
                 'symmetry_mean',]
outcome_var='diagnosis'
print(df.head())
x_train, x_test, y_train, y_test = train_test_split(df[predictor_var], df[outcome_var], random_state=0, test_size=0.3)

## 1. Fit a Logsitic Regression model with the specified hyperparameters
model = LogisticRegression(penalty='none', fit_intercept=True)
model.fit(x_train, y_train)
## 2. Fit the model to training data and obtain cofficient and intercept
print(model.coef_, model.intercept_)

## 3. Print accuracy, precision, recall and F1 score on test data

# Get the predictions
#
y_pred = model.predict(x_test)
#
# Calculate the confusion matrix
#
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

# F1 score
print(f1_score(y_test, y_pred))

# accuracy:
print(accuracy_score(y_test, y_pred))

# precision:
print(precision_score(y_test, y_pred))

# recall:
print(recall_score(y_test, y_pred))
```

## **Prediction Thresholds**

Logistic regression not only predicts the class of a sample, but also the probability of a sample belonging to each class. In this way, a measure of certainty is associated to each prediction. In the case of a binary classification, the default threshold value is 50% – predicted probabilities higher than this are associated to the positive class, lower are associated to the negative class. If two samples have predicted probabilities of 51% and 99%, both will be considered positive with the default threshold. However, if the threshold is increased to 60%, now the predicted probability of 51% will be assigned the negative class.

Consider the histogram of the predicted probabilities for the logistic regression classifier trained above. The benign (or negative class) is depicted in blue, the malignant (or positive class) in orange for the breast cancer data set. The benign cases are heavily clustered around zero, which is good as they will be correctly classified as benign, whereas malignant cases are heavily clustered around one. The vertical lines depict hypothetical threshold values at 25%, 50%, and 75%. For the highest threshold, almost all the samples above 75% belong to the malignant class, but there will be some benign cases that are misdiagnosed as malignant (false-positives). In addition, there are a number of malignant cases that are missed (false-negatives). If in stead the lowest threshold value is used, almost all the malignant cases are identified, but there are more false-positives.

Therefore, the value of the threshold is an additional lever that can be used to tune a model’s predicts – higher values are generally associated to lower false-positives/higher false-negatives, whereas a lower value is associated to lower false-negatives/higher false-positives.

## **ROC Curve and AUC**

We have examined how changing the threshold can affect the logistic regression predictions, without retraining or changing the coefficients of the model. In essence, there is a continuum of predictions available in a single model by the varying the threshold incrementally from zero to one. For each of these thresholds, the true and false-positive rates can be calculated and then plot. The resulting curve these points form is known as the Receiver Operating Characteristic (ROC) curve.

To plot the ROC curve, we use `scikit-learn`‘s `roc_curve` function, where the input contains the arrays `y_true` and `y_score` and output the arrays false-positive rate, true-positive rate, and threshold values. The plot of the true-positive rate vs false-positive rate gives us the ROC Curve.

We’ve plotted the ROC Curve for the dataset and model we’ve been working with throughout this lesson.You will notice that the threshold value is not discernible from the curve alone. We’ve labelled the threshold value on the plot itself for clarity, chose a list of ~5 points to label the threshold value on the curve. Plot the ROC curve for a “DummyClassifier” using the most-frequent class. We’ve also plotted the ROC curve of a “dummy classifier”, that predicts that all the data points belong to the more frequent class.

## **Class Imbalance**

Class imbalance is when your binary classes are not evenly split. Technically, anything different from a 50/50 distribution would be imbalanced and need appropriate care. However, the more imbalanced your data is, the more important it is to handle. In the case of rare events, sometimes the positive class can be less than 1% of the total.

The first issue that can arise is in randomly splitting your data – the smaller your dataset and more imbalanced the classes, the more likely there can be significant differences in classes for the training and test set. One way to mitigate this is to randomly split using stratification on the class labels. This ensures a nearly equal class distribution in your train and test sets.

In addition to using stratified sampling, class imbalance can be handled by undersampling the majority class or oversampling the minority class. For oversampling, repeated samples (with replacement) are taken from the minority class until the size is equal to that of the majority class. The downside is that the same data is used multiple times, giving a higher weight to these samples. On the other side, an undersampling used less of the majority class data to have equal size as the minority class. The downside here is that less data is used to build a model.

When training a model, the default setting is that every sample is weighted equally. However, in the case of class imbalance, this can result in poor predictive power in the smaller of the two classes. A way to counteract this in logistic regression is to use the parameter `class_weight='balanced'`. This applies a weight inversely proportional to the class frequently, therefore supplying higher weight to misclassified instances in the smaller class. While overall accuracy may not increase, this can increase the accuracy in the smaller class. Again in the breast cancer dataset, the most important classification is in the underrepresented malignant class.

```python
## 1. Stratified Sampling
x_train, x_test, y_train, y_test = train_test_split(df[predictor_var], df[outcome_var], random_state=6, test_size=0.3, stratify=df[outcome_var])
print('Train positivity rate: ')
print(sum(y_train)/y_train.shape[0])
print('Test positivity rate: ')
print(sum(y_test)/y_test.shape[0])

##2. Balanced Class weights
log_reg = LogisticRegression(penalty='none', max_iter=1000, fit_intercept=True, tol=0.000001, class_weight='balanced')
log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)
print(f'Recall Score: {recall_score( y_test, y_pred)}')
print(f'Accuracy Score: {accuracy_score( y_test, y_pred)}')

##3. Over/under-sampling
df_oversample = df[df[outcome_var]==1].sample(df[df[outcome_var]==0].shape[0], replace=True)
new_os_df = pd.concat([df[df[outcome_var]==0], df_oversample])
print('Oversampled class counts:')
print(new_os_df[outcome_var].value_counts())
```