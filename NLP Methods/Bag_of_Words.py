class bag_of_words:
  def __init__(self,min_freq=1):#only entertaining words with minimum no. of appearance=1
    self.vocabulary={}
    self.min_freq=min_freq

  def build_vocabulary(self,documents):
    from collections import Counter
    word_counts=Counter()
    freq={}#initilise 
    for doc in documents:
      tokens=re.findall(r'\b\w+\b',doc.lower())#lowecasing
      unique_tokens=set(tokens)#entertaining only unique tokens
      for token in unique_tokens:
        freq[token]=freq.get(token,0)+1#iterating over all 
        self.vocabulary = {}
    index = 0
    # Manually assign indices starting from 0
    for word, count in freq.items():
        if count >= self.min_freq:
            self.vocabulary[word] = index
            index += 1#indexing the tokens
    #verify the size and maximum index
    print(f"Built vocabulary size: {len(self.vocabulary)}")
    if self.vocabulary:
        max_index = max(self.vocabulary.values())
        print(f"Max index in vocabulary: {max_index}")
    self.vocabulary={word: i for i, (word, count) in enumerate(freq.items()) if count >= self.min_freq}

  def transformation(self,documents):#generating word vectors
    vectors=[]#initialise
    vocab_size = len(self.vocabulary)
    for doc in documents:
      tokens=re.findall(r'\b\w+\b',doc.lower())
      vector=np.zeros(len(self.vocabulary),dtype=int)
      for token in tokens:
        loc=self.vocabulary.get(token)
        if loc is not None: #and 0 <= loc < vocab_size:
          vector[loc]=vector[loc]+1
      vectors.append(vector)
    return np.array(vectors)

  def fitting(self,documents):
    self.build_vocabulary(documents)
    return self.transformation(documents)
