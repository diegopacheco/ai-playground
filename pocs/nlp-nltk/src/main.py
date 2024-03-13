import nltk

sentence = """At eight o'clock on Thursday morning
 Arthur didn't feel very good."""
print("sentence: " + str(sentence)) 

tokens = nltk.word_tokenize(sentence)
print("tokens: " + str(tokens)) 

tagged = nltk.pos_tag(tokens)
print("tags: " + str(tagged)) 

entities = nltk.chunk.ne_chunk(tagged)
print("entities: " + str(entities))