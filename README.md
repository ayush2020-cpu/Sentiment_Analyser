# Sentiment_Analyser
This is a simple sentiment analysis project that uses the Naive Bayes algorithm to classify movie reviews as either positive or negative. The model is trained on the movie_reviews dataset from NLTK and uses the CountVectorizer from sklearn to transform the text into feature vectors for classification.
Requirements

    Python 3.x
    nltk library
    scikit-learn library

Installation

    Clone or download this repository.

    Install the required dependencies:

    pip install nltk scikit-learn

    Download NLTK datasets by running the script, which will automatically download the movie_reviews and stopwords datasets the first time you run the code.

Usage

    The script loads movie reviews from NLTK's movie_reviews corpus.
    It preprocesses the data, converting it into feature vectors.
    A Naive Bayes classifier (MultinomialNB) is trained on the data.
    The model's accuracy is evaluated using the test set and printed.
    You can test the model by passing any text to the predict_sentiment() function, which will return whether the sentiment is positive or negative.

Example

sample_text = "I love this movie! It's fantastic."
print(f"Sentiment of '{sample_text}': {predict_sentiment(sample_text)}")

sample_text2 = "The movie was terrible and boring."
print(f"Sentiment of '{sample_text2}': {predict_sentiment(sample_text2)}")

Output

The program will output the accuracy of the model on the test data, followed by the sentiment of any test text you input.

Example output:

Accuracy: 81.40%
Sentiment of 'I love this movie! It's fantastic.': pos
Sentiment of 'The movie was terrible and boring.': neg

License

This project is licensed under the MIT License - see the LICENSE file for details.
