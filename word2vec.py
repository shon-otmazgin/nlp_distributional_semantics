import numpy as np


def load_and_normalize_vectors(file):
    W = []
    words = []
    with open(file, 'r', encoding='utf8') as f:
        for row in f:
            row = row.split()
            words.append(row[0])
            v = np.array(row[1:], dtype=np.float)
            W.append(v / np.linalg.norm(v))
    return np.array(W), np.array(words)


bow_vecs_file = 'word2vec/bow5.words'
dep_vecs_file = 'word2vec/deps.words'

bow_W, bow_words = load_and_normalize_vectors(bow_vecs_file)
dep_W, dep_words = load_and_normalize_vectors(dep_vecs_file)

print(f'bow5 words: {len(bow_words)}')
print(f'bow5 matrix shape: {bow_W.shape}')
print(f'dep words: {len(dep_words)}')
print(f'dep matrix shape: {dep_W.shape}')

# W and words are numpy arrays.
bow_w2i = {w:i for i,w in enumerate(bow_words)}
dep_w2i = {w:i for i,w in enumerate(dep_words)}

target_words = ['car', 'bus', 'hospital', 'hotel', 'gun', 'bomb', 'horse', 'fox', 'table', 'bowl', 'guitar', 'piano']
file = 'word2_vec_top20.txt'
with open(file, 'w', encoding='utf8') as f:
    for target in target_words:
        bow_v = bow_W[bow_w2i[target]] # get target vector
        dep_v = dep_W[dep_w2i[target]]  # get target vector

        bow_sims = bow_W.dot(bow_v)
        dep_sims = dep_W.dot(dep_v)

        most_similar_ids = bow_sims.argsort()[-1:-20:-1]
        bow_sim_words = bow_words[most_similar_ids]

        most_similar_ids = dep_sims.argsort()[-1:-20:-1]
        dep_sim_words = dep_words[most_similar_ids]

        f.write(f'{target}\n\n')
        for w1, w2 in zip(bow_sim_words, dep_sim_words):
            f.write(f"{w1:<20} {w2:<20}\n")
        f.write(f'*********\n')
