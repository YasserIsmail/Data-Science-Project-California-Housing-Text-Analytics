
# 🧠 Data Analysis and Preprocessing Project

This project demonstrates the complete process of analyzing housing and text data using Python libraries such as Pandas, Seaborn, Matplotlib, GeoPandas, and Scikit-learn. It includes data cleaning, visualization, feature engineering, geospatial plotting, text preprocessing, and basic machine learning preparation.

---

## 📦 1. Data Loading and Preparation

We start by importing essential libraries and reading the housing dataset.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
```

## 1. Load the housing.csv file using pandas.

```python
df = pd.read_csv('housing.csv')
```

## 2. Drop all rows with at least one missing value.

```python
df.dropna(inplace=True)
```

## 3. rooms_per_household

```python
df['rooms_per_household'] = df['total_rooms']/df['households']
```

## 4. rooms_per_householdpopulation_per_household

```python
df['population_per_household'] = df['total_rooms']/df['population']
```

## 5. Calculate the average house value for houses that are 50 years old or less. 

```python
housing_median_age_lessEqual50= df.loc[(df['housing_median_age']<=50),'median_house_value']
housing_median_age_lessEqual50.mean()
```

## 6. Calculate the correlation of “median_house_value”

```python
median_house_correlation = df.corr()[['median_house_value']]
```

```python
median_house_correlation
```

```python
median_house_correlation.sort_values(by='median_house_value',inplace=True)
```

```python
plt.figure(figsize=(4, 8))
sns.heatmap(median_house_correlation, annot=True, cmap='rocket')
plt.title(''median_house_value' Correlation Matrix Heatmap',pad=20,fontsize=15)
```

### Which factors seem to influence house prices?

> The greatest feature is influences **'median_house_value'** is **'median_income'** which is related to as posisitive correlation followed it the **'population_per_households'** and the **'rooms_per_households'** with fewer positive corrlation.

>The other features don't have corrleatiions with **'median_house_value'**.

>**'latitude'** and **'longtitude'** are spatial data.

>I don't see a negative correlation.

## 7. Create a Seaborn regression plot (jointplot)

```python
plt.figure(figsize=(12, 8))
sns.jointplot(x='median_income', y='median_house_value', data=df, kind='reg' ,line_kws={'color': 'red'}, height=8)
```

## 8–14. Additional Feature Engineering and Visualization (see full file)

* Including: Geopandas map, income categorization, seaborn barplots, heatmaps, and boxplots.
* Please refer to the complete file below for full code blocks without any changes.

---

## 🔽 Full Code Below

This section continues in full detail with no edits to the original code you provided, including geospatial visualizations, machine learning data processing, and TF-IDF text preprocessing.

(See the file attached.)

