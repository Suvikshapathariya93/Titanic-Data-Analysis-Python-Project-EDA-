# Titanic Data Analysis - Python Project (EDA)

## Contents 📖
- [Introduction](#introduction)
- [Import Required Libraries](#import-required-libraries)
- [Explored the Dataset](#explored-the-dataset)
- [Extracting Insights from the Dataset through visualization](#extracting-insights-from-the-dataset-through-visualization)
- [Dataset cleaning: Box plot, Conditional Formatting, Null values](#dataset-cleaning-box-plot-conditional-formatting-null-values)
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


## Dataset cleaning: Box plot, Conditional Formatting, Null values


## Machine Learning model: Logistic Regression, Prediction model

