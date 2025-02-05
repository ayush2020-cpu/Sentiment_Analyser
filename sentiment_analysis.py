import nltk
import string
from nltk.corpus import stopwords, movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

nltk.download('movie_reviews')
nltk.download('stopwords')

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = ''.join([char.lower() for char in text if char not in string.punctuation])
    return ' '.join([word for word in text.split() if word not in stop_words])

texts = [preprocess_text(' '.join(doc)) for doc, _ in documents]
labels = [category for _, category in documents]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

accuracy = metrics.accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy * 100:.2f}%")

def predict_sentiment(text):
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text])
    prob = model.predict_proba(text_vectorized).max()
    return model.predict(text_vectorized)[0] if prob > 0.5 else "Can't predict the emotion"

while True:
    user_input = input("Enter a sentence (type 'stop' to exit): ")
    if user_input.lower() == 'stop':
        break
    print(f"Sentiment: {predict_sentiment(user_input)}")
