# Twitter Sentiment Analysis of Public Opinion on Pfizer Vaccines

### Project Summary

This project presents an end-to-end machine learning pipeline for analyzing public sentiment towards the Pfizer vaccine using Twitter data. The model classifies tweets into **positive**, **negative**, and **neutral** categories to provide insights into public perception. The analysis involves comprehensive data preprocessing, feature engineering, and a comparative study of multiple classification algorithms to identify the most accurate model for this task.

---

### Project Workflow and Methodology

1.  **Data Cleaning and Preprocessing**

    The initial dataset of tweets was processed to prepare it for analysis. This critical phase involved several steps to clean the unstructured text data:
    * **Text Normalization**: Standardized the text by converting it to lowercase and removing irrelevant characters, URLs, mentions, and hashtags.
    * **Tokenization**: Broke down each tweet into individual words or tokens.
    * **Stopword Removal**: Filtered out common English stopwords (e.g., "the", "a", "is") to reduce noise in the dataset.
    * **Stemming**: Applied the Porter Stemmer algorithm to reduce words to their root form (e.g., "vaccinating" -> "vaccin"), ensuring that different forms of a word were treated as a single entity.

2.  **Exploratory Data Analysis (EDA) and Feature Engineering**

    To understand the dataset's characteristics, I performed an initial sentiment polarity analysis using `TextBlob`. This helped in labeling the tweets for supervised learning. The distribution of sentiments was visualized to check for class balance.

    For feature engineering, the cleaned text was transformed into a numerical representation using a **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorizer. This technique captures the relative importance of each word in the corpus, providing meaningful features for the machine learning models.

3.  **Machine Learning Model Implementation and Comparison**

    Several classification models were trained on the TF-IDF features to predict the sentiment of a given tweet. The models implemented for comparison include:
    * **Logistic Regression**
    * **Support Vector Classifier (SVC)**
    * **Naive Bayes**

    To optimize the performance of the Support Vector Classifier, **GridSearchCV** was employed for hyperparameter tuning, systematically finding the best combination of parameters like `kernel`, `C`, and `gamma`.

4.  **Model Evaluation and Results**

    Each model's performance was rigorously evaluated on a held-out test set using standard classification metrics: **Accuracy**, **Precision**, **Recall**, and **F1-Score**. A **confusion matrix** was generated for each model to visualize its performance in distinguishing between the positive, negative, and neutral classes.

    The final analysis identified the most effective model based on a holistic review of these metrics, demonstrating a robust solution for automated sentiment classification from social media data.