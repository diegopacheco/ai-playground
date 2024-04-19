import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the vader_lexicon
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

text = "I love coding in Zig. It is a great language!"
print(text.ljust(60, ' ') 
      + str(analyze_sentiment(text)))

print("I HATE SAFE! ".ljust(60, ' ') + str(analyze_sentiment("I HATE SAFE!")))
print("Vanilla ice cream is ok. ".ljust(60, ' ') + str(analyze_sentiment("Vanilla ice cream is ok")))