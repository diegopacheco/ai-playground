from transformers import pipeline

en_fr_translator = pipeline("translation_en_to_fr")
result = en_fr_translator("How old are you?")
print(f"English to French translation: {result}")