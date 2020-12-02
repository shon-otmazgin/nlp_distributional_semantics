import time
from collections import defaultdict, Counter
from itertools import groupby
from itertools import combinations
import math

ID = 'ID'
FORM = 'FORM'
LEMMA = 'LEMMA'
CPOSTAG = 'CPOSTAG'
POSTAG = 'POSTAG'
FEATS = 'FEATS'
HEAD = 'HEAD'
DEPREL = 'DEPREL'
PHEAD = 'PHEAD'
PDEPREL = 'PDEPREL'
FREQ = '#freq'
FIELDS_H = [ID, FORM, LEMMA, CPOSTAG, POSTAG, FEATS, HEAD, DEPREL, PHEAD, PDEPREL]

# https://webapps.towson.edu/ows/ptsspch.htm#:~:text=Content%20words%20are%20words%20that,language%20as%20they%20become%20obsolete.
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
ADJECTIVES = ['JJ', 'JJR', 'JJS']
ADVERBS = ['RB', 'RBR', 'RBS']

CONTENT_WORD_TAGS = NOUNS + VERBS + ADJECTIVES + ADVERBS

THRESHOLD = 100
FEATURE_FREQUENT = 2
target_words = ['car', 'bus', 'hospital', 'hotel', 'gun', 'bomb', 'horse', 'fox', 'table', 'bowl', 'guitar', 'piano']


def read_data(file):
    word_frequency = Counter()
    counts = defaultdict(Counter)
    with open(file, 'r', encoding='utf8') as f:
        sents_dep = [list(group) for k, group in groupby(f.readlines(), lambda x: x == '\n') if not k]

    for s_p in sents_dep:
        s_tokenize = [{h: v for h, v in zip(FIELDS_H, token.split())} for token in s_p]
        content_words = [d[LEMMA] for d in s_tokenize if d[POSTAG] in CONTENT_WORD_TAGS]
        for w in content_words:
            word_frequency[w] += 1
        for w, w_c in combinations(content_words, 2):
            counts[w][w_c] += 1
            counts[w_c][w] += 1
    return counts, word_frequency


file = 'wikipedia.tinysample.trees.lemmatized'
counts, word_frequency = read_data(file=file)
z = sum([sum(counts[w].values()) for w in counts])


def p(u=None, att=None):
    if u is None and att is None:
        return 1
    if u is None:
        return sum(counts[att].values()) / z
    if att is None:
        return sum(counts[u].values()) / z
    return counts[u][att] / z


def get_similarities(target_word, counts):
    similarity_result = Counter()

    attributes = counts[target_word]
    for att in attributes:
        for v in counts[att]:
            u_att = math.log(p(u=target_word, att=att) / (p(u=target_word) * p(att=att)))
            v_att = math.log(p(u=v, att=att) / (p(u=v) * p(att=att)))
            if u_att <= 0 or v_att <= 0:
                continue
            similarity_result[v] += u_att * v_att

    print(similarity_result.most_common())


if __name__ == '__main__':
    start_time = time.time()

    file = 'counts_words.txt'
    with open(file, 'w') as f:
        f.writelines([f"{w[0]} {w[1]}\n" for w in word_frequency.most_common(n=50)])

    #check p(u, att) = 1
    p_1 = sum([p(u=w, att=att) for w in counts for att in counts[w]])
    print(p_1)

    # check p(u) = 1
    p_2 = sum([p(u=w) for w in counts])
    print(p_2)

    # check p(u, att) = 1
    p_3 = sum([p(att=att) for att in counts])
    print(p_3)

    get_similarities(target_word='gun', counts=counts)
    print("--- %s seconds ---" % (time.time() - start_time))