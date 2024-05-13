import numpy as np
from chainer import Chain, Variable, optimizers, iterators, training
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class SkipGram(Chain):
    def __init__(self, n_vocab, n_units):
        super(SkipGram, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.out = L.Linear(n_units, n_vocab)

    def __call__(self, x, contexts):
        e = self.embed(contexts)
        shape = e.shape
        x = F.broadcast_to(x[:, None], (shape[0], shape[1]))
        e = F.reshape(e, (shape[0] * shape[1], self.out.W.shape[1]))
        x = F.reshape(x, (shape[0] * shape[1],))
        center_predictions = self.out(e)
        loss = F.softmax_cross_entropy(center_predictions, x)
        return loss

# Dummy dataset
train = [('the', 'quick'), ('quick', 'brown'), ('brown', 'fox'), ('fox', 'jumps')]

# Create a dictionary mapping words to integers
word2index = {word: i for i, word in enumerate(set(word for words in train for word in words))}

# Convert the training data to integers
train = [(word2index[word], word2index[context]) for word, context in train]

# Size of the vocabulary
n_vocab = len(word2index)

# Setup model and optimizer
model = SkipGram(n_vocab, 100)  # 100 is the size of word embeddings
optimizer = optimizers.Adam()
optimizer.setup(model)

# Setup iterator
train_iter = iterators.SerialIterator(train, batch_size=2, repeat=True)

# Setup updater and trainer
updater = training.updaters.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (20, 'epoch'), out='result')

# Add extensions
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'elapsed_time']))

# Run the trainer
trainer.run()