import keras_nlp
import tensorflow_datasets as tfds

imdb_train, imdb_test = tfds.load(
  "imdb_reviews",
  split=["train", "test"],
  as_supervised=True,
  batch_size=16,
)

# Load a BERT model.
classifier = keras_nlp.models.BertClassifier.from_preset("bert_base_en_uncased", num_classes=2)

# Fine-tune on IMDb movie reviews.
classifier.fit(imdb_train, validation_data=imdb_test)

# Predict two new examples.
print(classifier.predict(["What an amazing movie!", "A total waste of my time."]))