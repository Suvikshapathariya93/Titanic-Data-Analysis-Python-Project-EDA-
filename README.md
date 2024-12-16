# Titanic Data Analysis - Python Project (EDA)

## Contents 📖
- [Introduction](#introduction)
- [Import Required Libraries](#import-required-libraries)
- [Explored the Dataset](#explored-the-dataset)
- [Extracting Insights from the Dataset through visualization](#extracting-insights-from-the-dataset-through-visualization)
- [EDA: Box plot, Conditional Formatting, Null values](#EDA-box-plot-conditional-formatting-null-values)
- [Machine Learning model: Logistic Regression, Prediction model](#machine-learning-model-logistic-regression-prediction-model)

## Introduction

**🚢 Titanic Dataset:** Data Analysis
The Titanic dataset is one of the most famous datasets in data science, often used for beginners to practice exploratory data analysis (EDA) and machine learning. This project focuses on analyzing the dataset to uncover key insights about the passengers and their survival. By leveraging tools like Python, Pandas, and visualization libraries, we aim to answer interesting questions and provide a comprehensive view of the data.

**The goal of this project is to:**

Explore and preprocess the Titanic dataset.
Identify patterns and trends in passenger demographics and survival rates.
Visualize key insights using various Python libraries.
Highlight important factors that influenced survival, such as gender, age, and class.

**Tools Used🛠️:**
- Language : Python
- Library : Pandas, Numpy, Matplotlib, Seaborn
- IDE : Google colab

## Import Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

```
- **NumPy** (```np```): Used for numerical computations and working with arrays.
- **Pandas** (```pd```): Ideal for data manipulation and analysis, particularly with tabular data.
- **Matplotlib** (```plt```): A foundational library for creating static, animated, and interactive plots.
- **Seaborn** (```sns```): Built on Matplotlib, it simplifies creating aesthetically pleasing and informative statistical graphics.

## Explored the Dataset

```python
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
```
- ```pd.read_csv```: Reads a CSV file from the provided URL into a Pandas DataFrame.
- **Titanic Dataset**: This dataset contains information about passengers on the Titanic, often used for data analysis and machine learning projects and dataset collected from kaggle.
- ``df``: The variable df holds the loaded data as a DataFrame, allowing for easy manipulation and analysis.

```python
#exploring titanic data
df.head(3)
```

- ```df.head(n)```: Displays the first n rows of the DataFrame (default is 5 if n is not provided).
- **Purpose**: In this case, ```df.head(3)``` shows the first 3 rows of the Titanic dataset, giving a quick preview of the data structure and its contents.
- ```df.tail(n)```: Display the last n rows of the DataFrame (default is 5 if n is not provided).
- ```df.ndim```: Display the dimension of the dataset
- ```df.columns```: Display the columns name with datatype
- ```df.info()```: Display the Range index, total column counts, index of columns, column names, Non-null counts, data types of columns and memory use.
- ```df.isnull()```: Display the null value in TRUE and FALSE form of each and every entity.
- ```df.isnull().sum()```: Display the sum of null value by columns.
- ```df.describe()```: Display the statistics calculation according to our dataset like count, mean, standard devaition, min, max etc. 

## Extracting Insights from the Dataset through visualization

1. Visualize the null value through HeatMap.
```python
#This code visualizes the missing data in the dataset, making it easy to identify which columns or rows have null (missing) values and their distribution.
sns.heatmap(df.isnull())
plt.show()
```
- ```sns.heatmap```: Creates a heatmap visualization using the Seaborn library.
- ```df.isnull()```: Generates a DataFrame of the same shape as df, where each cell is True if the value is missing (NaN) and False otherwise.
**The heatmap highlights missing data in the dataset, where True values are usually represented by brighter colors and False values by darker colors.**
- ```plt.show()```: Displays the heatmap.

2. Visualize the survival distribution across genders.
```python
#Purpose :This code visualizes the survival distribution across genders, helping to identify gender-based survival trends in the Titanic dataset.
sns.catplot(x='Survived', col='Sex', kind='count', data=df)
plt.show()
```
- ```sns.catplot```: Creates a categorical plot using the Seaborn library.
- ```x='Survived'```: Sets the x-axis to the "Survived" column, showing survival status (0 = Did not survive, 1 = Survived).
- ```col='Sex'```: Separates the plot into two subplots, one for each gender ("Male" and "Female").
- ```kind='count'```: Displays a count plot, showing the frequency of each survival status for each gender.
- ```data=df```: Uses the Titanic dataset stored in the DataFrame df.
- ```plt.show()```: Displays the generated plots.

3. Visualize the survival distribution across passenger classes.
```python
#Purpose : Visualizes the survival distribution across passenger classes to highlight the influence of class on survival.
sns.countplot(x='Survived', hue='Pclass', data=df)
```
- ```sns.countplot```: Creates a count plot using the Seaborn library.
- ```x='Survived'```: Sets the x-axis to the "Survived" column, showing survival status (0 = Did not survive, 1 = Survived).
- ```hue='Pclass'```: Adds color coding by "Pclass" (Passenger Class: 1 = First, 2 = Second, 3 = Third), allowing differentiation by passenger class.
- ```data=df```: Uses the Titanic dataset stored in the DataFrame df.

4. Visualize the survival distribution acorss different genders and passenger classes together.
```python
#Purpose: This code visualizes the survival distribution across different genders and passenger classes. It helps identify patterns, such as whether survival rates were influenced by gender or passenger class.
sns.catplot(x='Survived', col='Sex', hue='Pclass', kind='count', data=df)
```
- ```sns.catplot```: Creates a categorical plot using the Seaborn library.
- ```x='Survived'```: Sets the x-axis to the "Survived" column, showing survival status (0 = Did not survive, 1 = Survived).
- ```col='Sex'```: Separates the plot into two subplots, one for each gender ("Male" and "Female").
- ```hue='Pclass'```: Adds color coding by "Pclass" (Passenger Class: 1 = First, 2 = Second, 3 = Third) to show survival distribution by class within each gender.
- ```kind='count'```: Displays a count plot, showing the frequency of each survival status for each class and gender combination.
- ```data=df```: Uses the Titanic dataset stored in the DataFrame df.
- ```plt.show()```: Displays the generated plot.

5. Visualize the age distribution of passengers in the dataset as a Histogram.
```python
#Purpose: This code visualizes the age distribution of passengers in the dataset as a histogram. It provides insights into the age demographics of passengers, such as the most common age groups.
sns.displot(df['Age'].dropna(), kde=False, color='darkgreen', bins=40)
plt.show()
```
- ```sns.displot```: Creates a distribution plot using the Seaborn library.
- ```df['Age']```: Specifies the "Age" column from the dataset to analyze the distribution of ages.
- ```.dropna()```: Removes any missing values (NaN) from the "Age" column to ensure only valid data is plotted.
- ```kde=False```: Disables the Kernel Density Estimate (KDE) curve, leaving only the histogram to represent the data.
- ```color='darkgreen'```: Sets the color of the bars in the histogram to dark green.
- ```bins=40```: Divides the range of ages into 40 equal intervals (bins), ensuring a detailed view of the distribution.
- ```plt.show()```: Displays the plot.

6. Visualize the number of passengers with varying counts of Siblings/Spouse aboard.
```python
#This count plot provides a visual summary of the number of passengers with varying counts of siblings/spouses aboard. It helps analyze the family size distribution among passengers.
sns.countplot(x = 'SibSp', data = df)
```
- ```sns.countplot```: Creates a count plot using the Seaborn library.
- ```x='SibSp'```: Specifies the "SibSp" column on the x-axis, which represents the number of siblings and spouses each passenger had aboard the Titanic.
- ```data=df```: Specifies the dataset (df) as the source for the plot.
- ```plt.show()```: Displays the generated plot.

## EDA: Box plot, Conditional Formatting, Null values

1. Exploratory data analysis (EDA) for investigating relationships between variables like age and passenger class.
```python
#This plot provides insights into the age demographics of passengers within each class. For instance, it helps determine if certain classes had younger or older passengers on average.
plt.figure(figsize=(12,6))
sns.boxplot(x = 'Pclass', y = 'Age', data = df, palette = 'icefire')
```
- ```plt.figure(figsize=(12,6))```: Sets the figure size to 12 inches wide and 6 inches tall, ensuring the plot is well-proportioned and easy to read.
- ```sns.boxplot```: Creates a box plot using the Seaborn library.
- ```x='Pclass'```: Specifies the "Pclass" (Passenger Class) column for the x-axis, representing classes (1 = First, 2 = Second, 3 = Third).
- ```y='Age'```: Specifies the "Age" column for the y-axis, showing the age distribution for each class.
- ```data=df```: Uses the dataset df as the source for the plot.
- ```palette='icefire'```: Applies the "icefire" color palette to the box plot for aesthetic styling.
- ```plt.show()```: Displays the final plot.
- **Shows statistical information such as:**
  - Median age for each class.
  - Interquartile range (IQR), which highlights the middle 50% of the data.
  - Outliers (ages significantly above or below the usual range).
 
2. Apply conditional formatting to fill null values.
```python
#This function is designed to impute missing values in the "Age" column of the dataset based on the passenger class (Pclass).
def  impute_age(cols):
  Age = cols[0]
  Pclass = cols[1]
  if pd.isnull(Age):
    if Pclass == 1:
      return 37
    elif Pclass == 2:
      return 29
    else : 
      return 20
  else: 
   return Age
```
- Input:
  - cols: A pair of values — Age (the passenger's age) and Pclass (the passenger class: 1, 2, or 3).
- Logic:
  - If the Age is missing ```(pd.isnull(Age))```, the function assigns an age based on the passenger class:
    - Class 1: Replace missing age with 37 (assumed median age for 1st class passengers).
    - Class 2: Replace missing age with 29.
    - Class 3: Replace missing age with 20.
    - If the Age is not missing, the function returns the original Age value.
- Output: The function returns either the imputed age (based on class) or the existing age.

- **Why Is This Useful?**
   - Missing values in numerical columns like Age can cause issues in data analysis or machine learning models.
   - Imputing missing values based on a related feature like Pclass ensures that the replacements are more accurate and meaningful compared to a single overall average.
    - For example, 1st-class passengers may generally be older, while 3rd-class passengers may have younger ages, so this approach considers class-wise trends.

3. Apply above function
```python
df['Age'] = df[['Age', 'Pclass']].apply(impute_age, axis = 1)
```
- This line applies the ```impute_age``` function to the DataFrame ```df``` and replaces the missing values in the "Age" column.
- It uses the values of the "Age" and "Pclass" columns as inputs to decide the replacement for missing ages.

4. Drop the unwanted columns
```python
df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)
```
- ```df.drop()```: The drop() function is used to remove specific rows or columns.
- ```['Sex', 'Embarked', 'Name', 'Ticket']```: This is the list of column names that are being removed.
- ```axis=1```: Specifies that the operation is performed on columns.
- ```inplace=True```: Modifies the DataFrame in place, meaning no new DataFrame is created, and the original df is updated.

## Machine Learning model: Logistic Regression, Prediction model
1. Split the data for model in into test and train
```python
# Building a machine learning model
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.drop('Survived', axis =1), df['Survived'], test_size = 0.30, random_state = 101)
```
- ```train_test_split()```: A function from sklearn used to split the data into training and testing subsets.
- ```df.drop('Survived', axis=1)```: Removes the target column (Survived) from the DataFrame to create the feature matrix (X).
- ```df['Survived']```: Extracts the target variable (Survived) from the DataFrame.
- ```test_size=0.30```: Specifies that 30% of the data will be used for testing, while the remaining 70% will be used for training.
- ```random_state=101```: Ensures that the split is reproducible. The same split will occur every time the code is run with this random_state.

- **Why This Line Is Useful**:
    - Training Set (X_train, Y_train): Used to train the machine learning model.
    - Testing Set (X_test, Y_test): Used to evaluate the model's performance on unseen data.
    - Ensures that the "Survived" column (target) is isolated for prediction purposes, while other columns are used as features.

2. Feature scaling
```python
#This code standardizes the features in the training and testing datasets to have a mean of 0 and a standard deviation of 1, a crucial step in preprocessing for machine learning models.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
```
- ```StandardScaler```: A preprocessing tool from sklearn that standardizes data by removing the mean and scaling to unit variance.
- Create the Scaler Object (sc):
  - ```sc``` = ```StandardScaler()```: Initializes the scaler.
- Fit and Transform the Training Data:
  - ```sc.fit_transform(X_train)```: Computes the mean and standard deviation of the X_train features.
  - Standardizes the training data: The transformed ```X_train``` now has a mean of 0 and a standard deviation of 1 for each feature.
- Transform the Testing Data:
  - ```sc.transform(X_test)```: Uses the mean and standard deviation from the training data to standardize X_test.
                                Ensures consistency between training and testing datasets.
- Print Scaled Training Data: ```print(X_train)``` displays the standardized X_train dataset.

- **Why This Line Is Useful:**
    - Standardization Benefits:
      - Helps models (like logistic regression, SVMs, or neural networks) perform better by ensuring features are on the same scale.
      - Avoids bias caused by features with large magnitudes dominating others.
    - Training-Testing Consistency: Ensures that the same scaling is applied to both the training and testing sets for reliable model evaluation.
   
3. Training the model
```python
#This code creates and trains a logistic regression model using the training dataset.
Logistic regression is a classification algorithm used to predict discrete outcomes, such as whether a passenger survived or not in the Titanic dataset.
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 101)
model.fit(X_train, Y_train)
```
- Import Logistic Regression:
  - ```from sklearn.linear_model import LogisticRegression```: Imports the logistic regression model from scikit-learn's linear models library.
- Create the Model:
  - ```model = LogisticRegression(random_state=101)```: Initializes a logistic regression model.
  - The ```random_state``` parameter ensures the process is reproducible when randomness is involved (e.g., in solvers or regularization).
- Train the Model:
  - ```model.fit(X_train, Y_train)```: Fits the logistic regression model to the training data.
  - ```X_train```: The feature set (independent variables).
  - ```Y_train```: The target variable (dependent variable, e.g., survival status).
- What Happens During Training:
  - The logistic regression model learns the relationship between the features in X_train and the target variable Y_train.
  - It optimizes the coefficients for each feature to best predict the target outcome.

- **Why This Line Is Useful:**
  - Logistic regression is simple yet powerful for binary classification tasks (like predicting survival in the Titanic dataset).
  - By training the model, it is ready to make predictions on new data (e.g., testing data).

- **Output:**
  - The model object is trained and contains the learned parameters (coefficients and intercepts) that can be used to make predictions. 

4. Prediction with confusion matrix
```python
from sklearn.metrics import confusion_matrix, accuracy_score
```
- This code imports two evaluation metrics from sklearn to measure the performance of a machine learning model:
  - ```confusion_matrix```: Provides a detailed breakdown of the model's predictions versus actual outcomes.
  - ```accuracy_score```: Calculates the overall accuracy of the model's predictions.

- **Explanation**
  
a. ```confusion_matrix```: A table that summarizes the performance of a classification model by showing the number of:
  - True Positives (TP): Correctly predicted positives.
  - True Negatives (TN): Correctly predicted negatives.
  - False Positives (FP): Incorrectly predicted as positive.
  - False Negatives (FN): Incorrectly predicted as negative.
b. ```accuracy_score```: Measures the proportion of correct predictions out of the total predictions made by the model:
  - ```accuracy = correct prediction/ Total prediction```
  - Gives a single metric to evaluate the model's overall performance.

- **Why This Line Is Useful:**
  - Confusion Matrix:
    - Provides detailed insight into the types of errors the model makes, which is especially useful for imbalanced datasets.
    - Helps identify if the model is biased toward one class.
  - Accuracy Score:
    - Offers a quick way to understand how well the model performs overall.
    - Easy to compute and interpret for balanced datasets.
   
- **Output:**
  - ```Confusion Matrix```: A matrix showing the count of true and false predictions for each class.
  - ```Accuracy Score```: A numeric value between 0 and 1 (or 0% to 100%) representing overall accuracy.

```python
#This code uses a trained machine learning model to make predictions on the testing dataset and evaluates the predictions using a confusion matrix.
predictions = model.predict(X_test)

accuracy = confusion_matrix(Y_test, predictions)
accuracy

accuracy = accuracy_score(Y_test, predictions)
accuracy
```
- **Making Predictions**: Using the model to predict the target values for the test dataset.
- **Confusion Matrix**: Generating a matrix to analyze the performance by comparing actual and predicted outcomes.
- **Accuracy Score**: Calculating the overall percentage of correct predictions.

- **Why This Line Is Useful:**
  - Confusion Matrix: Provides detailed insight into the types of predictions (correct vs incorrect) made by the model.
  - Accuracy Score: Offers a quick and intuitive measure of the model’s performance as a percentage or decimal.


**I'm open to feedback and discussions! Reach out with any questions, suggestions, or collaboration ideas.** [LinkedIn | Suviksha Pathariya](https://www.linkedin.com/in/suviksha-pathariya/)

**Don’t forget to star ⭐ and follow this repository if you found it useful!**

**Stay tuned for more features and updates coming soon!**
