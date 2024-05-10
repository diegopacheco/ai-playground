import mxnet as mx
import gluonnlp as nlp

glove = nlp.embedding.create('glove', source='glove.6B.50d')

def cos_similarity(embedding, word1, word2):
    vec1, vec2 = embedding[word1], embedding[word2]
    return mx.nd.dot(vec1, vec2) / (vec1.norm() * vec2.norm())

print('Similarity between "baby" and "infant": ', cos_similarity(glove, 'baby', 'infant').asnumpy()[0])
