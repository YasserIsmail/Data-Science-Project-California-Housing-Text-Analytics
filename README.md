
# 🧠 Data Science Project: California Housing & Text Analytics

This project showcases a full data science pipeline including data cleaning, feature engineering, geospatial analysis, machine learning preprocessing, and natural language processing using real-world datasets. The work demonstrates Python-based workflows, visualizations, and transformations, suitable for portfolio presentation.

---

## 📂 Overview

This repository includes:

1. **Data Cleaning and Feature Engineering**
2. **Statistical and Visual Analysis**
3. **Geospatial Mapping**
4. **Machine Learning Pipeline**
5. **Text Mining and NLP**

---

## 1. 🧹 Data Cleaning & Feature Engineering

```python
import pandas as pd

df = pd.read_csv("housing.csv")
df.dropna(inplace=True)
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["population_per_household"] = df["population"] / df["households"]
```

---

## 2. 📊 Statistical and Visual Analysis

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation
correlations = df.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False)

# Income vs House Value
sns.jointplot(data=df, x="median_income", y="median_house_value", kind="reg")
```

![Income vs House Value](plots/cell_output_0.png)

```python
# Geo Scatter Plot
plt.scatter(df["longitude"], df["latitude"],
            s=df["population"]/100, c=df["median_house_value"],
            cmap="viridis", alpha=0.5)
```

![Geospatial Scatter Plot](plots/cell_output_1.png)

---

## 3. 🗺 Geospatial Mapping

```python
import geopandas as gpd
from shapely.geometry import Point

df["geometry"] = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
geo_df = gpd.GeoDataFrame(df, geometry="geometry")
```

![California Map Plot](plots/cell_output_2.png)

---

## 4. 🤖 Machine Learning Preprocessing

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = ["ocean_proximity"]

preprocessor = ColumnTransformer(transformers=[
    ("num", MinMaxScaler(), numeric_features),
    ("cat", OneHotEncoder(), categorical_features)
])
```

```python
# Boxplot for Outlier Detection
sns.boxplot(data=df[["total_rooms", "population"]])
```

![Boxplot for Outlier Detection](plots/cell_output_6.png)

---

## 5. 🔤 Text Mining and NLP

```python
import json

df_sarcasm = pd.read_json("sarcasm.json", lines=True)
df_sarcasm["domain_name"] = df_sarcasm["article_link"].apply(lambda x: x.split("/entry")[0])
df_sarcasm["domain_name"].value_counts().plot(kind="bar")
```

![Headline Count per Domain](plots/cell_output_7.png)

```python
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

df_sarcasm["cleaned"] = df_sarcasm["headline"].apply(preprocess)
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df_sarcasm["cleaned"])
```

![TF-IDF Example Output](plots/cell_output_8.png)

---

## 📌 Summary

- Python data pipeline with real estate and text datasets
- Exploratory data visualization
- Spatial data processing with GeoPandas
- Machine learning preprocessing using scikit-learn
- Natural Language Processing with NLTK and TF-IDF

