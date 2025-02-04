import nltk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

nltk.download('movie_reviews')
nltk.download('stopwords')

from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

texts = [' '.join(doc) for doc, category in documents]
labels = [category for doc, category in documents]

vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'))
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    return model.predict(text_vectorized)[0]

sample_text = "I love this movie! It's fantastic."
print(f"Sentiment of '{sample_text}': {predict_sentiment(sample_text)}")

sample_text2 = "The movie was terrible and boring."
print(f"Sentiment of '{sample_text2}': {predict_sentiment(sample_text2)}")
