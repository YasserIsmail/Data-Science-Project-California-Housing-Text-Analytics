# 🏠 California Housing Data Analysis

This project provides a comprehensive analysis and visualization of the California housing dataset using Python, including:
- Data cleaning and feature engineering
- Statistical summaries and correlations
- Geospatial visualizations with GeoPandas
- Outlier detection and removal
- Machine learning preprocessing


### 📦 **Import Libraries**

```python
# Importing pandas for data manipulation and analysis (tables, DataFrames)
import pandas as pd

# Importing numpy for numerical computations and working with arrays
import numpy as np

# Importing seaborn for enhanced data visualization (built on matplotlib)
import seaborn as sns

# Importing matplotlib for plotting and visualizing data (line charts, histograms, etc.)
import matplotlib.pyplot as plt

# Importing geopandas for working with geospatial data (e.g., shapefiles, GeoJSON)
import geopandas as gpd
````

> These libraries form the core stack for data analysis, visualization, and geospatial processing in Python.
>
> * `pandas` and `numpy`: essential for handling structured and numerical data.
> * `seaborn` and `matplotlib`: used to create clear and insightful visualizations.
> * `geopandas`: enables spatial analysis using vector data like points, lines, and polygons.


### 📂 **Load the `housing.csv` file**

```python
# Load the housing dataset into a pandas DataFrame
df = pd.read_csv('housing.csv')
```

> This command reads the `housing.csv` file and stores it in the `df` variable as a DataFrame.  
> The DataFrame allows for structured analysis, filtering, and visualization of the housing data.


### 🧹 **Drop Missing Values**

```python
# Remove all rows that contain any missing values
df.dropna(inplace=True)
```

> This command removes any row in the DataFrame that contains at least one missing (NaN) value.  
> The `inplace=True` parameter ensures the change is applied directly to the original DataFrame.

### 🧮 **Compute rooms per household**

```python
# Add a new column that calculates average rooms per household
df['rooms_per_household'] = df['total_rooms'] / df['households']
```

> This feature helps measure housing density by dividing the total number of rooms by the number of households.  
> It is useful for understanding residential space allocation.

---

### 🧮 **Compute population per household**

```python
# Add a new column that calculates population per household (note: the original formula uses total_rooms)
df['population_per_household'] = df['total_rooms'] / df['population']
```

> This line computes a ratio, but it's likely mislabeled — dividing `total_rooms` by `population` actually gives rooms per person, not population per household.  
> To get the true population per household, use:  
> `df['population_per_household'] = df['population'] / df['households']`


### 📊 **Average house value for homes <= 50 years**

```python
# Filter house values for homes that are 50 years old or less
housing_median_age_lessEqual50 = df.loc[(df['housing_median_age'] <= 50), 'median_house_value']

# Calculate the average (mean) house value for these filtered homes
housing_median_age_lessEqual50.mean()
```

> This section calculates the mean value of houses that are 50 years old or younger.  
> It's useful for analyzing how age affects housing prices.

---

### 🔗 **Calculate correlation with house value**

```python
# Compute correlation of all columns with 'median_house_value'
median_house_correlation = df.corr()[['median_house_value']]
```

> This command generates a correlation matrix that shows how each feature is related to `median_house_value`.

```python
# Display the correlation values
median_house_correlation
```

> This displays the correlation table in the output.

```python
# Sort the correlation values in ascending order
median_house_correlation.sort_values(by='median_house_value', inplace=True)
```

> Sorting helps quickly identify which variables have the strongest positive or negative relationship with house value.

```python
# Plot the sorted correlation values as a heatmap
plt.figure(figsize=(4, 8))
sns.heatmap(median_house_correlation, annot=True, cmap='rocket')
plt.title('\'median_house_value\' Correlation Matrix Heatmap', pad=20, fontsize=15)
```

> This heatmap visually highlights the strength and direction of correlation for each feature with `median_house_value`.

---

### 🖊 **Comment on Correlations**

> Use this section to analyze and interpret the correlation values.  
> Which features are most positively or negatively correlated with house value?  
> For example:  
> - High correlation: `median_income`  
> - Negative correlation: `latitude`, `longitude`  
> These insights guide feature selection in predictive modeling.

### 📈 **Regression Plot: Income vs House Value**

```python
# Create a regression jointplot to visualize income vs house value
plt.figure(figsize=(12, 8))
sns.jointplot(x='median_income', y='median_house_value', data=df, kind='reg', line_kws={'color': 'red'}, height=8)
```

> This plot shows how `median_income` is linearly related to `median_house_value` using a regression line.

---

### 🗺 **Scatterplot with Spatial Reference**

```python
# Load California boundary shapefile and set CRS to EPSG:4326
gdf = gpd.read_file('ca_state.zip')
gdf.to_crs(4326, inplace=True)
```

> Load spatial boundary data and reproject it to a standard geographic coordinate system.

```python
# Plot housing data as spatially-referenced points over California
axis = gdf.plot(figsize=(10,12), color='lightgray', edgecolor='black', alpha=0.3)
sns.scatterplot(
    ax=axis,
    data=df,
    x='longitude',
    y='latitude',
    hue='median_house_value',
    size='population',
    palette='coolwarm',
    edgecolor='black',
    sizes=(20,250),
    alpha=0.6,
)
axis.set_title('Median House Value and Population', fontsize=16)
axis.set_xlabel('Longitude', fontsize=12)
axis.set_ylabel('Latitude', fontsize=12)
plt.legend(fontsize=12, bbox_to_anchor=(1, 1), loc='upper left')
```

> Overlay a geospatial scatterplot of housing data on the California map, with house value and population as visual indicators.

---

### 💰 **Categorize Income**

```python
# Compute income quantiles for binning
q1 = df['median_income'].quantile(0.25)
q2 = df['median_income'].quantile(0.5)
q3 = df['median_income'].quantile(0.75)
at95 = df['median_income'].quantile(0.95)
```

> These thresholds are used to categorize households based on income distribution.

```python
# Create a new column for income categories
df['income_cat'] = pd.Series(dtype='object')
```

> Initialize an empty column to store income class labels.

```python
# Assign income categories based on income quantiles
df.loc[df['median_income'] < q1, 'income_cat'] = 'Low'
df.loc[(df['median_income'] >= q1) & (df['median_income'] < q2), 'income_cat'] = 'Below average'
df.loc[(df['median_income'] >= q2) & (df['median_income'] < q3), 'income_cat'] = 'Above average'
df.loc[(df['median_income'] >= q3) & (df['median_income'] < at95), 'income_cat'] = 'High'
df.loc[df['median_income'] >= at95, 'income_cat'] = 'Very high'
```

> Label each household based on income quartile.

---

### 📊 **Histogram of Income Category by Ocean Proximity**

```python
# Show distribution of income categories by ocean proximity
plt.figure(figsize=(12, 8))
sns.histplot(data=df.sort_values(by='median_income'), x='income_cat', hue='ocean_proximity', multiple='dodge')
plt.grid(True, alpha=0.6, axis='y')
```

> Explore how income categories differ by geographic proximity to the ocean.

---

### 📉 **Bar Plot: Income Category vs House Value**

```python
# Compare house values by income category
plt.figure(figsize=(12, 8))
plt.grid(True, alpha=0.6, axis='y')
sns.barplot(x='income_cat', y='median_house_value', data=df.sort_values(by='median_income'))
```

> Visualize average housing prices across different income levels.

---

### 🌊 **Bar Plot: Ocean Proximity vs House Value**

```python
# Compare house values by proximity to the ocean
plt.figure(figsize=(12, 8))
sns.barplot(x='ocean_proximity', y='median_house_value', data=df)
plt.grid(True, alpha=0.6, axis='y')
```

> Evaluate how location near the ocean affects housing prices.

---

### 🔥 **Heatmap: Income Category vs Ocean Proximity**

```python
# Create a pivot table of average house values by income category and ocean proximity
heatmap_df = df[['ocean_proximity','income_cat','median_house_value']]
groups = heatmap_df.groupby(['ocean_proximity','income_cat'])

df_list = []
for i in groups.groups:
    df_list.append([i[0], i[1], groups.get_group(i)['median_house_value'].mean()])

heatmap_df = pd.DataFrame(df_list, columns=['ocean_proximity','income_cat','median_house_value'])
heatmap_df = heatmap_df.pivot('income_cat', 'ocean_proximity', 'median_house_value')
heatmap_df.drop(columns=['ISLAND'], inplace=True)
heatmap_df.sort_values(by=['<1H OCEAN','INLAND','NEAR BAY','NEAR OCEAN'], inplace=True)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_df, annot=True, fmt=".0f", linewidths=0.5)
```

> A detailed view of how both income and ocean proximity affect house values.

---

### 🔁 **Reload Data and Fill Missing Values**

```python
# Reload the dataset and handle missing values
df = pd.read_csv('housing.csv')
mean = df['total_bedrooms'].mean()
df['total_bedrooms'] = df['total_bedrooms'].fillna(mean)

median = df['total_rooms'].median()
df['total_rooms'] = df['total_rooms'].fillna(median)
```

> Reimport data and fill missing values using mean or median.  
> (KNN imputer is optional and commented out.)

---

### 📦 **Identify Outliers**

```python
# Select numeric columns and generate boxplots
num_df = df.select_dtypes(np.number)

plt.figure(figsize=(15, 8))
ncols = 3
num_features = len(num_df.columns)
nrows = (num_features + 1) // ncols

for i, column in enumerate(num_df.columns, 1):
    plt.subplot(nrows, ncols, i)
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')

plt.tight_layout()
plt.show()
```

> Use boxplots to visually inspect potential outliers in numeric data.

---

### ❌ **Remove Outliers with IQR Rule**

```python
# Function to drop outliers using 1.5*IQR method
def drop_outliers(df):
    columns = df.select_dtypes(np.number)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df.loc[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

df_droped_outlier = drop_outliers(df)
```

> Identify and eliminate extreme outliers using interquartile range.

```python
# Boxplots to compare before/after outlier removal
nrows = len(num_df.columns)
fig, axes = plt.subplots(nrows, 2, figsize=(15, 30))

for i, column in enumerate(num_df.columns):
    sns.boxplot(x=df[column], ax=axes[i, 0]) 
    axes[i, 0].set_title(f'{column}', fontsize=14, fontweight='bold')
    sns.boxplot(x=df_droped_outlier[column], ax=axes[i, 1])  
    axes[i, 1].set_title(f'{column}', fontsize=14, fontweight='bold')

fig.text(0.25, 0.95, 'Left: Original Data', ha='center', va='center', fontsize=16, fontweight='bold')
fig.text(0.75, 0.95, 'Right: Data with Outliers Removed', ha='center', va='center', fontsize=16, fontweight='bold')
plt.tight_layout()
fig.subplots_adjust(top=0.92, hspace=0.5)
```

> Visualize and compare the impact of removing outliers across all numeric features.

---

### 🧠 **Machine Learning Preprocessing**

```python
# Import preprocessing and transformation tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
```

> Required tools for splitting, normalizing, and encoding data before modeling.

```python
# Prepare features and target
X = df.drop(columns=['longitude','latitude','median_house_value']) 
y = df['median_house_value']
```

> Remove non-informative or spatial columns and isolate target variable.

```python
# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

> Use 70/30 split for training and evaluation.

```python
# Initialize transformers
scaler = MinMaxScaler()
encoder = OneHotEncoder(handle_unknown='ignore')
```

> Set up normalization for numeric data and encoding for categoricals.

```python
# Identify numeric and categorical features
num = X.select_dtypes(np.number)
cat = X.select_dtypes(exclude=[np.number])
```

> Separate columns for applying appropriate transformations.

```python
# Build a column transformer pipeline
processor = ColumnTransformer([
    ('scaler', scaler, num.columns),
    ('encoder', encoder, cat.columns)
])
```

> Combine both scaling and encoding into a unified preprocessing pipeline.

```python
# Apply transformations to training and test data
processed_X_train = processor.fit_transform(X_train)
processed_X_test = processor.transform(X_test)
processed_y_train = scaler.fit_transform(np.array(y_train.to_numpy().reshape(-1,1)))
processed_y_test  = scaler.transform(y_test.to_numpy().reshape(-1,1))
```

> Finalize transformed data ready for machine learning models.

---
