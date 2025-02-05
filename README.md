Sentiment Analysis Using Naive Bayes
Overview

This project performs sentiment analysis on movie reviews. It uses the Naive Bayes algorithm to predict whether a given sentence or movie review is positive or negative.
What does it do?

    Preprocesses movie reviews (removes stopwords, punctuation, and makes everything lowercase).
    Trains a machine learning model using a dataset of movie reviews.
    Predicts whether a given text has a positive or negative sentiment.

How to Use
1. Install Required Libraries

Before running the script, make sure you have the necessary Python libraries installed.

Run this command in your terminal or command prompt to install them:

pip install nltk scikit-learn

2. Run the Script

Save the script as sentiment_analysis.py, and then run it using the following command:

python sentiment_analysis.py

3. Input Sentences

After running the script, you can start typing sentences, and it will tell you whether the sentiment is positive or negative. If the sentiment is unclear, it will say "Can't predict the emotion".

To stop the program, type stop and press Enter.
Example:

Enter a sentence (type 'stop' to exit): I love this movie! It's amazing.
Sentiment: positive

Enter a sentence (type 'stop' to exit): The movie was really bad.
Sentiment: negative

4. Accuracy

The model’s accuracy will also be printed at the start, telling you how well it performs on the given movie review dataset.
