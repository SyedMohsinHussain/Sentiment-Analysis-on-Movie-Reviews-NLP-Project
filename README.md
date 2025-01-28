# Sentiment Analysis on IMDB Dataset

This project demonstrates sentiment analysis using machine learning techniques, specifically focusing on text classification. The primary objective is to build a sentiment classifier that predicts the sentiment of movie reviews based on labeled data.

## Requirements

This project requires the following libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `nltk`
- `sklearn`

To install the necessary libraries, run the following command:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn


## Dataset
The project uses multiple datasets for sentiment classification:

IMDB Dataset (train): Contains labeled movie reviews with sentiments (positive/negative).
Small Test Dataset: A smaller subset of reviews for testing the model.
Medium Test Dataset: A medium-sized dataset for evaluating the model's performance.
Large Test Dataset: A larger dataset that tests the model on more diverse data.
The datasets are CSV files and should be in the following format:

IMDB Dataset: Contains 'review' (text) and 'sentiment' (0 for negative, 1 for positive).
Test Datasets: Contain 'text' (reviews) and 'label' (sentiment).

# Installation
Clone or download the repository.
Place the dataset CSV files in the /content/ directory, or modify the paths in the code to point to your local dataset location.

How to Run
Run the code in a Python environment that supports Jupyter Notebooks or any IDE that supports Python.
The script will load the training and test datasets.
The text reviews will be preprocessed by removing unnecessary characters, converting to lowercase, and applying lemmatization.
The sentiment of reviews will be predicted using a Logistic Regression model.
The model will be evaluated on various datasets with performance metrics (Accuracy, Precision, Recall, and F1-score).
A bar chart will display the performance comparison across the training and test datasets.

## Functions
preprocess_text(text): Preprocesses the text by converting to lowercase, removing non-alphabetical characters, and lemmatizing words.
predict_sentiment(user_input): Accepts a user input (review), transforms it, and returns the sentiment prediction (0 or 1).
Performance Evaluation
The project evaluates model performance on the following metrics:

Accuracy: The proportion of correct predictions.
Precision: The ratio of correct positive predictions to the total predicted positives.
Recall: The ratio of correct positive predictions to the total actual positives.
F1 Score: The harmonic mean of Precision and Recall.
These metrics are calculated for each dataset (training, small test, medium test, and large test).

## Future Work
Implement more advanced models (e.g., Deep Learning models).
Experiment with different text preprocessing techniques such as stemming or removing stopwords.
Extend the dataset to include more diverse reviews and languages.
Conclusion
This project demonstrates a basic sentiment analysis workflow using machine learning techniques. The model can be further improved with more data and advanced algorithms.


## License

Notes:
- Be sure to update the file paths for datasets if they're not in the `/content/` directory.
- You can also update the license section based on your preferred open-source license.
