from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

app = Flask(__name__)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment using VADER
def analyze_sentiment_vader(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

# Function to analyze sentiment using TextBlob
def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        sentiment_vader = analyze_sentiment_vader(text)
        sentiment_textblob = analyze_sentiment_textblob(text)

        # Determine sentiment result
        if sentiment_vader > 0:
            sentiment_vader_result = "Positive"
        elif sentiment_vader < 0:
            sentiment_vader_result = "Negative"
        else:
            sentiment_vader_result = "Neutral"

        if sentiment_textblob > 0:
            sentiment_textblob_result = "Positive"
        elif sentiment_textblob < 0:
            sentiment_textblob_result = "Negative"
        else:
            sentiment_textblob_result = "Neutral"

        return render_template("index.html", sentiment_vader_result=sentiment_vader_result,
                               sentiment_textblob_result=sentiment_textblob_result,
                               sentiment_vader=sentiment_vader, sentiment_textblob=sentiment_textblob,
                               text=text)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
