Here's your content formatted and cleaned up for a **README.md** file:

````markdown
# 🎉 Sentiment Analysis on OxygenSelfCare Application Reviews 

## 📊 Overview
This project performs sentiment analysis on user reviews of the **OxygenSelfCare** mobile application. Using the **Support Vector Machine (SVM)** algorithm, we classify reviews into positive and negative sentiments based on textual data. The process involves data scraping, preprocessing, and analysis, followed by model training and evaluation.

## 📚 Table of Contents
- [Introduction]
- [Requirements]
- [Data Collection]
- [Data Preprocessing]
- [Sentiment Analysis]
- [Model Training]
- [Evaluation]
- [Wordcloud Visualization]
- [Conclusion]

## 🧠 Introduction
This project aims to classify user feedback into positive and negative sentiments using reviews from the **OxygenSelfCare** application. The reviews are collected via scraping, processed for cleaning, tokenization, and vectorization, and then classified using a **Support Vector Machine (SVM)**.

## 💻 Requirements
Make sure you have the following libraries installed:
- Python 3.x 🐍
- pandas 📊
- sklearn 📚
- matplotlib 📈
- wordcloud 🌈
- seaborn 📊
- google-play-scraper 📲
- nltk 📖

Install the necessary libraries using the following command:

```bash
pip install -r requirements.txt
````

## 📱 Data Collection

Data for this project is scraped from the Google Play Store using the `google-play-scraper` package. The review data includes:

* Review ID 📝
* Username 💬
* Rating ⭐
* Review Text 💭
* Date 📅

The following code is used to scrape the reviews:

```python
from google_play_scraper import app, reviews

# Scrape reviews
app_id = "com.oxygenselfcare.app"
reviews_data = reviews(app_id)
```

## 🧹 Data Preprocessing

Data preprocessing involves several key steps:

1. **Cleaning**: Remove special characters, URLs, and unnecessary symbols.
2. **Case Folding**: Convert all text to lowercase.
3. **Tokenization**: Split the text into words.
4. **Stopword Removal**: Remove common words (like 'the', 'is', etc.) that do not contribute to sentiment.
5. **Stemming**: Reduce words to their root form using `nltk.stem` library.

The preprocessing steps are implemented in the code as follows:

```python
def remove_urls(tweet):
    # Function to remove URLs from text
    return re.sub(r'http\S+', '', tweet)

def remove_stopwords(tweet):
    # Function to remove stopwords
    stopwords = set(STOPWORDS)
    tokens = tweet.split()
    return ' '.join([word for word in tokens if word not in stopwords])
```

## 💬 Sentiment Analysis

Applying sentiment classification based on a predefined positive and negative lexicon:

```python
def determine_sentiment(text):
    positive_count = sum(1 for word in text.split() if word in positive_lexicon)
    negative_count = sum(1 for word in text.split() if word in negative_lexicon)
    if positive_count > negative_count:
        return "Positive" 👍
    else:
        return "Negative" 👎
```

## 📈 Model Training

The project uses **Support Vector Machine (SVM)** to classify sentiments. The SVM model is trained using the following steps:

1. **Vectorization**: Convert text data into numerical format using TF-IDF vectorizer.
2. **Train-Test Split**: Split the data into training and testing sets.
3. **Model Fitting**: Fit the SVM model to the training data.
4. **Prediction**: Use the trained model to predict sentiments on the test set.

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## ✅ Evaluation

The model's performance is evaluated using metrics like accuracy, precision, recall, and F1-score:

```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
```

The confusion matrix visualizes the number of **True Positives**, **False Positives**, **True Negatives**, and **False Negatives**.

## 🌈 Wordcloud Visualization

To gain insights from the most frequent terms in the reviews, a **word cloud** is generated:

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(background_color="white", width=800, height=400).generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```

## 🎯 Conclusion

This project successfully implemented sentiment analysis on user reviews of the **OxygenSelfCare** app using **Support Vector Machine (SVM)**. The data preprocessing steps, including cleaning, tokenization, stopword removal, and stemming, proved effective in preparing the data for analysis. The **SVM** model performed well in classifying sentiment, with good evaluation metrics (accuracy, precision, recall, F1-score).
The **wordcloud** visualization provided insights into the most frequently mentioned terms, offering valuable feedback for app improvement.
