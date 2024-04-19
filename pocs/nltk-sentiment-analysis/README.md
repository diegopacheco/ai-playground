### Result

* Sentiment Analysis using VADER
* the 4 scores are: positive, negative, neutral and compound
* The compound score is a metric that calculates the sum of all the lexicon ratings which have been normalized between -1(most extreme negative) and +1 (most extreme positive).

```
❯ ./run.sh
[nltk_data] Downloading package vader_lexicon to
[nltk_data]     /home/diego/nltk_data...
[nltk_data]   Package vader_lexicon is already up-to-date!
I love coding in Zig. It is a great language!               {'neg': 0.0, 'neu': 0.411, 'pos': 0.589, 'compound': 0.8622}
I HATE SAFE!                                                {'neg': 0.579, 'neu': 0.0, 'pos': 0.421, 'compound': -0.2714}
Vanilla ice cream is ok.                                    {'neg': 0.0, 'neu': 0.645, 'pos': 0.355, 'compound': 0.296}
```