#Loading Dependencies
from collections import defaultdict, Counter
from scipy.sparse import lil_matrix
import numpy as np
import math

class TFIDFVectorizer:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.vocab = {}
        self.idf = {}

    def fit(self, documents):
        doc_freq = defaultdict(int)
        total_docs = len(documents)
        for doc in documents:
            tokens = set(doc.split())  #unique tokens per doc
            for token in tokens:
                doc_freq[token] += 1
        self.vocab = {
            token: idx for idx, (token, freq) in enumerate(doc_freq.items())
            if freq >= self.min_freq
        }
        for token in self.vocab:
            df = doc_freq[token]
            self.idf[token] = math.log((1 + total_docs) / (1 + df)) + 1  #smooth IDF
#taking log t reduce the difference between smaller and larger frequency words
    def transform(self, documents):
        rows = len(documents)
        cols = len(self.vocab)
        X = lil_matrix((rows, cols), dtype=np.float32)

        for i, doc in enumerate(documents):
            tf = Counter(doc.split())
            total_terms = sum(tf.values())
            for token, count in tf.items():
                if token in self.vocab:
                    tf_val = count / total_terms
                    idf_val = self.idf[token]
                    X[i, self.vocab[token]] = tf_val * idf_val
        return X.tocsr()

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)
