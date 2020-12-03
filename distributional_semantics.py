import time
from collections import defaultdict, Counter
from itertools import groupby
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
FIELDS_H = [ID, FORM, LEMMA, CPOSTAG, POSTAG, FEATS, HEAD, DEPREL, PHEAD, PDEPREL]

# https://webapps.towson.edu/ows/ptsspch.htm#:~:text=Content%20words%20are%20words%20that,language%20as%20they%20become%20obsolete.
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
ADJECTIVES = ['JJ', 'JJR', 'JJS']
ADVERBS = ['RB', 'RBR', 'RBS']

CONTENT_WORD_TAGS = NOUNS + VERBS + ADJECTIVES + ADVERBS
target_words = ['car', 'bus', 'hospital', 'hotel', 'gun', 'bomb', 'horse', 'fox', 'table', 'bowl', 'guitar', 'piano']


class WordsStats:
    SENTENCE = 'sentence'
    WINDOW = 'window'
    DEPENDENCY = 'dependency'
    WORD_FREQUENT = 100
    FEATURE_WORD_FREQUENT = 75
    FEATURES_LIMIT = 100

    def __init__(self, window):
        self.window = window
        self.word_counts = defaultdict(lambda: defaultdict(Counter))   # word_counts[method][word][feature] = #freq
        self.total = {self.SENTENCE: 0, self.WINDOW: 0, self.DEPENDENCY: 0}
        self.total_u = {self.SENTENCE: defaultdict(lambda: 0), self.WINDOW: defaultdict(lambda: 0), self.DEPENDENCY: defaultdict(lambda: 0)}
        self.total_att = {self.SENTENCE: defaultdict(lambda: 0), self.WINDOW: defaultdict(lambda: 0), self.DEPENDENCY: defaultdict(lambda: 0)}
        self.word_frequency = Counter()

    def fit(self, file):
        with open(file, 'r', encoding='utf8') as f:
            sents_dep = [list(group) for k, group in groupby(f.readlines(), lambda x: x == '\n') if not k]

        for s_d in sents_dep:
            s_tokenize = [{h: v for h, v in zip(FIELDS_H, token.rstrip('\n').split('\t'))} for token in s_d]
            self.words_co_occurring(sentence_tokenized=s_tokenize)

        self.filter_stats(method=self.SENTENCE)
        self.filter_stats(method=self.WINDOW)

        for method in self.word_counts:
            for w, w_c in self.word_counts[method].items():
                # cache p(*,*)
                self.total[method] += sum(w_c.values())

                # cache p(u,*)
                self.total_u[method][w] += sum(w_c.values())

                # cache p(*,att)
                for att, att_c in w_c.items():
                    self.total_att[method][att] += att_c

        return self

    def filter_stats(self, method):
        """
        ### Filtering Features:
        ### feature is a tuple of (word, att1(optional), att1(optional), ...)
        ### 100 most_common features for a word
        ### frequent of the feature's word (location 0 in the tuple) should be grater than 75
        """
        for w in self.word_counts[method]:
            c = Counter()
            for att, count in self.word_counts[method][w].most_common(n=self.FEATURES_LIMIT):
                if self.word_frequency[att[0]] >= self.FEATURE_WORD_FREQUENT:
                    c[att] = count
            self.word_counts[method][w] = c

    def words_co_occurring(self, sentence_tokenized):

        content_words = [d[LEMMA] for d in sentence_tokenized if d[POSTAG] in CONTENT_WORD_TAGS]
        for i, w in enumerate(content_words):
            low = i - self.window if i >= self.window else 0
            high = i + self.window + 1 if i + self.window + 1 <= len(content_words) else len(content_words)
            words_window = content_words[low:i] + content_words[i+1:high]
            for co_word in words_window:
                self.word_counts[self.WINDOW][w][(co_word,)] += 1

            sent_window = content_words[0:i] + content_words[i+1:]
            for co_word in sent_window:
                self.word_counts[self.SENTENCE][w][(co_word,)] += 1

            self.word_frequency[w] += 1


class WordSimilarities:
    def __init__(self, stats: WordsStats):
        self.stats = stats
        self.l2_norm = defaultdict(lambda: defaultdict(lambda: 0.))
        self.word_vecs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))) # for efficient cosine similarity
        self.att_vecs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))  # for efficient cosine similarity

    def fit(self):
        for method in self.stats.word_counts:
            p_all, p_u, p_att = 0, 0, Counter()
            for w, counter in self.stats.word_counts[method].items():
                for att in counter:
                    if att not in p_att:
                        p_att[att] += self.p(att=att, method=method)
                    p_all += self.p(u=w, att=att, method=method)
                p_u += self.p(u=w, method=method)
            print(f'method: {method} p1: {p_all} p1: {p_u} p1: {sum(p_att.values())}')

        # pre process - create ppmi vector to each word and also calc the norm.
        for method in self.stats.word_counts:
            for w in self.stats.word_counts[method]:
                if self.stats.word_frequency[w] < self.stats.WORD_FREQUENT:
                    continue
                for att in self.stats.word_counts[method][w]:
                    ppmi = self.get_PPMI(u=w, att=att, method=method)
                    self.l2_norm[method][w] += ppmi**2
                    self.word_vecs[method][w][att] = ppmi
                    self.att_vecs[method][att][w] = ppmi
                self.l2_norm[method][w] = math.sqrt(self.l2_norm[method][w])
        return self

    def get_PPMI(self, u, att, method):
        ppmi = math.log(self.p(u=u, att=att, method=method) / (self.p(u=u, method=method) * self.p(att=att, method=method)))
        return ppmi if ppmi > 0 else 0

    def p(self, method, u=None, att=None):
        if u is None and att is None:
            return 1
        if u is None:
            return self.stats.total_att[method][att] / self.stats.total[method]
        if att is None:
            return self.stats.total_u[method][u] / self.stats.total[method]
        return self.stats.word_counts[method][u][att] / self.stats.total[method]

    def get_similarities(self, target_word, method):
        similarity_result = Counter()

        for att, u_att_ppmi in self.word_vecs[method][target_word].items():
            for v, v_att_ppmi in self.att_vecs[method][att].items():
                similarity_result[v] += (u_att_ppmi * v_att_ppmi)

        for v in similarity_result:
            similarity_result[v] /= (self.l2_norm[method][target_word] * self.l2_norm[method][v])
        del similarity_result[target_word]
        return similarity_result


if __name__ == '__main__':

    start_time = time.time()
    file = 'wikipedia.sample.trees.lemmatized'
    stats = WordsStats(window=2).fit(file=file)
    print(f'Finished fit stats {(time.time() - start_time):.3f} sec')

    file = 'counts_words.txt'
    with open(file, 'w') as f:
        f.writelines([f"{w[0]} {w[1]}\n" for w in stats.word_frequency.most_common(n=50)])

    start_time = time.time()
    word_sim = WordSimilarities(stats=stats).fit()
    print(f'Finished fit Similarities {(time.time() - start_time):.3f} sec')

    print()
    for word in target_words:
        start_time = time.time()
        sent_similarities = word_sim.get_similarities(target_word=word, method=stats.SENTENCE)
        window_similarities = word_sim.get_similarities(target_word=word, method=stats.WINDOW)
        print(f'inference for {word} {(time.time() - start_time):.3f} sec')
        print(word)
        for (sent_word, sent_sim), (win_word, win_sim) in zip(sent_similarities.most_common(20), window_similarities.most_common(20)):
            print(f"{sent_word:<10} {sent_sim:.3f}\t{win_word:<10} {win_sim:.3f}")
        print('*********')
