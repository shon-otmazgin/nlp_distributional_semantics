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

WORD_FREQUENT = 100
FEATURE_WORD_FREQUENT = 75
FEATURES_LIMIT = 100
target_words = ['car', 'bus', 'hospital', 'hotel', 'gun', 'bomb', 'horse', 'fox', 'table', 'bowl', 'guitar', 'piano']


class WordsStats:
    def __init__(self):
        self.word_counts = defaultdict(Counter)  # dict of words to dict of attributes count
        self.att_counts = defaultdict(Counter)  # dict of att to dict of words count
        self.word_frequency = Counter()
        self.total = 0

    def fit(self, file, method='window', window_size=None):
        with open(file, 'r', encoding='utf8') as f:
            sents_dep = [list(group) for k, group in groupby(f.readlines(), lambda x: x == '\n') if not k]

        for s_d in sents_dep:
            s_tokenize = [{h: v for h, v in zip(FIELDS_H, token.rstrip('\n').split('\t'))} for token in s_d]
            if method == 'window':
                self.words_co_occurring(window=window_size, sentence_tokenized=s_tokenize)
            elif method == 'dependency':
                pass

        for w in self.word_counts:
            c = Counter()
            for att, count in self.word_counts[w].most_common(n=FEATURES_LIMIT):
                if self.word_frequency[att[0]] >= FEATURE_WORD_FREQUENT:
                    c[att] = count
                    self.att_counts[att][w] = count
            self.word_counts[w] = c

        self.total = sum([self.word_counts[w][att] for w in self.word_counts for att in self.word_counts[w]])
        return self

    def words_co_occurring(self, sentence_tokenized, window=None):
        # window = None -> all the sentence

        content_words = [d[LEMMA] for d in sentence_tokenized if d[POSTAG] in CONTENT_WORD_TAGS]

        for i, w in enumerate(content_words):
            if window is None:
                low, high = 0, len(content_words)
            else:
                if i < window:
                    low = 0
                else:
                    low = i - window
                if i + window + 1 > len(content_words):
                    high = len(content_words)
                else:
                    high = i + window + 1
            words_window = content_words[low:i]
            words_window += content_words[i + 1:high]
            for co_word in words_window:
                self.word_counts[w][(co_word, )] += 1
                # self.att_counts[(co_word,)][w] += 1

            self.word_frequency[w] += 1


class WordSimilarities:
    def __init__(self, stats: WordsStats, corpus_threshold, feature_threshold):
        self.stats = stats
        self.corpus_threshold = corpus_threshold
        self.feature_threshold = feature_threshold
        self.l2_norm = Counter()
        self.word_vecs = defaultdict(Counter)
        self.att_vecs = defaultdict(Counter)

    def fit(self):
        #check p(u, att) = 1
        p_1 = sum([self.p(u=w, att=att) for w in self.stats.word_counts for att in self.stats.word_counts[w]])
        print(f"p(u, att) = {p_1}")

        # # check p(u) = 1
        p_2 = sum([self.p(u=w) for w in self.stats.word_counts])
        print(f"p(u) = {p_2}")

        # # check p(att) = 1
        p_3 = sum([self.p(att=att) for att in self.stats.att_counts])
        print(f"p(att) = {p_3}")

        for w in self.stats.word_counts:
            for att in self.stats.word_counts[w]:
                ppmi = self.get_PPMI(u=w, att=att)
                self.l2_norm[w] += ppmi**2
                self.word_vecs[w][att] = ppmi
                self.att_vecs[att][w] = ppmi
            self.l2_norm[w] = math.sqrt(self.l2_norm[w])
        return self

    def get_PPMI(self, u, att):
        ppmi = math.log(self.p(u=u, att=att) / (self.p(u=u) * self.p(att=att)))
        return ppmi if ppmi > 0 else 0

    def p(self, u=None, att=None):
        if u is None and att is None:
            return 1
        if u is None:
            return sum(self.stats.att_counts[att].values()) / self.stats.total
        if att is None:
            return sum(self.stats.word_counts[u].values()) / self.stats.total
        return self.stats.word_counts[u][att] / self.stats.total

    def get_similarities(self, target_word):
        similarity_result = Counter()

        for att in stats.word_counts[target_word]:
            for v in stats.att_counts[att]:
                similarity_result[v] += (self.word_vecs[target_word][att] * self.att_vecs[att][v])

        for v in similarity_result:
            similarity_result[v] /= (self.l2_norm[target_word] * self.l2_norm[v])
        return similarity_result


if __name__ == '__main__':

    start_time = time.time()
    file = 'wikipedia.sample.trees.lemmatized'
    stats = WordsStats().fit(file=file, method='window', window_size=2)
    print(f'Finished fit stats {time.time() - start_time}')

    file = 'counts_words.txt'
    with open(file, 'w') as f:
        f.writelines([f"{w[0]} {w[1]}\n" for w in stats.word_frequency.most_common(n=50)])

    start_time = time.time()
    word_sim = WordSimilarities(stats=stats, corpus_threshold=WORD_FREQUENT, feature_threshold=FEATURE_WORD_FREQUENT).fit()
    print(f'Finished fit Similarities {time.time() - start_time}')

    print()
    for word in target_words:
        start_time = time.time()
        word_similarities = word_sim.get_similarities(target_word=word)
        print(f'Finished Similarities for {word} {time.time() - start_time}')
        print(word_similarities.most_common(5))
        print()

    print("--- %s seconds ---" % (time.time() - start_time))