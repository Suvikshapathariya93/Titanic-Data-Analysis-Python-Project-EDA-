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
 


