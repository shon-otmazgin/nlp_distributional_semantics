import time
from collections import defaultdict, Counter
from itertools import groupby
import math
from utils import *


class WordsStats:
    def __init__(self, window, word_freq, attributes_word_freq, attributes_limit):
        self.window = window
        self.word_frequency = Counter()
        self.word_counts = defaultdict(lambda: defaultdict(Counter))   # word_counts[method][word][attribute] = #freq
        self.total = {SENTENCE: 0, WINDOW: 0, DEPENDENCY: 0}
        self.total_w = {SENTENCE: defaultdict(lambda: 0), WINDOW: defaultdict(lambda: 0), DEPENDENCY: defaultdict(lambda: 0)}
        self.total_att = {SENTENCE: defaultdict(lambda: 0), WINDOW: defaultdict(lambda: 0), DEPENDENCY: defaultdict(lambda: 0)}
        self.str2int = defaultdict(lambda: 0)
        self.int2str = {}
        self.word_freq = word_freq
        self.attributes_word_freq = attributes_word_freq
        self.attributes_limit = attributes_limit

    def _get_w_hash(self, w):
        hashed_w = self.str2int[w]
        if hashed_w == 0:
            hashed_w = hash(w)
            self.str2int[w] = hashed_w
            self.int2str[hashed_w] = w

        return hashed_w

    def _get_w(self, hashed_w):
        return self.int2str[hashed_w]

    def fit(self, file):
        with open(file, 'r', encoding='utf8') as f:
            sentences = [list(group) for k, group in groupby(f.readlines(), lambda x: x == '\n') if not k]

        for sent in sentences:
            tokenized_sent = [{h: v for h, v in zip(FIELDS_H, token.rstrip('\n').split('\t'))} for token in sent]
            self.words_co_occurring(tokenized_sentence=tokenized_sent)
            # self.words_dependency(sentence_tokenized=tokenized_sent)

        self.filter_stats()

        for method in self.word_counts:
            for w, w_c in self.word_counts[method].items():
                # cache p(*,*)
                self.total[method] += sum(w_c.values())
                # cache p(u,*)
                hashed_w = self._get_w_hash(w)
                self.total_w[method][hashed_w] += sum(w_c.values())

                # cache p(*,att)
                for att, att_c in w_c.items():
                    hashed_att = self._get_w_hash(att)
                    self.total_att[method][hashed_att] += att_c

        return self

    # def words_dependency(self, sentence_tokenized):
    #     content_words, prp_words, noun_words = [], [], []
    #     for w in sentence_tokenized:
    #         if w[POSTAG] in CONTENT_WORD_TAGS:
    #             content_words.append(w)
    #         elif w[POSTAG] in PREPOSITION:
    #             prp_words.append(w)
    #         if w[POSTAG] in NOUNS:
    #             noun_words.append(w)
    # 
    #     for w in content_words:
    #         w_head = sentence_tokenized[int(w[HEAD])-1] if int(w[HEAD]) > 0 else None
    #         if w_head is None:
    #             continue
    # 
    #         hashed_w = self._get_w_hash(w[LEMMA])
    #         # case 1
    #         for prp in prp_words:
    #             if prp[HEAD] == w[ID]:
    #                 for noun in noun_words:
    #                     if noun[HEAD] == prp[ID]:
    #                         label = w[DEPREL] + " " + prp[LEMMA]
    # 
    #                         feature1 = (noun[LEMMA], label, self.OUT)
    #                         feature2 = (w[LEMMA], label, self.IN)
    # 
    #                         hashed_noun = self._get_w_hash(noun[LEMMA])
    #                         self.word_counts[self.DEPENDENCY][hashed_w][feature1] += 1
    #                         self.word_counts[self.DEPENDENCY][hashed_noun][feature2] += 1
    # 
    #         if w_head[POSTAG] in PREPOSITION: # case 2
    #             prp_head = sentence_tokenized[int(w_head[HEAD])-1]
    #             label = w_head[DEPREL] + " " + w_head[LEMMA]
    # 
    #             feature1 = (prp_head[LEMMA], label, self.IN)
    #             feature2 = (w[LEMMA], label, self.OUT)
    # 
    #             hashed_prp = self._get_w_hash(prp_head[LEMMA])
    #             self.word_counts[self.DEPENDENCY][hashed_w][feature1] += 1
    #             self.word_counts[self.DEPENDENCY][hashed_prp][feature2] += 1
    # 
    #         else:
    #             self.word_counts[self.DEPENDENCY][hashed_w][(w_head[LEMMA], w[DEPREL], self.IN)] += 1
    #             hashed_w_head = self._get_w_hash(w_head[LEMMA])
    #             self.word_counts[self.DEPENDENCY][hashed_w_head][(w[LEMMA], w[DEPREL], self.OUT)] += 1

    def filter_stats(self):
        """
        Filtering Features:
        attribute is a tuple of (word, att1(optional), att1(optional), ...)
        100 most_common features for a word
        frequent of the attribute's word (location 0 in the tuple) should be grater than 75
        """
        for method in self.word_counts:
            for w in self.word_counts[method]:
                c = Counter()
                hashed_w = self._get_w_hash(w)
                for hashed_att, count in self.word_counts[method][hashed_w].most_common(n=self.attributes_limit):
                    att = self._get_w(hashed_att)
                    hashed_w_att = self._get_w_hash(att[0])
                    if self.word_frequency[hashed_w_att] >= self.attributes_word_freq:
                        c[hashed_att] = count
                self.word_counts[method][hashed_w] = c

    def words_co_occurring(self, tokenized_sentence):
        content_words = [row[LEMMA] for row in tokenized_sentence if row[POSTAG] in CONTENT_WORD_TAGS]
        for i, w in enumerate(content_words):
            hashed_w = self._get_w_hash(w)

            low = i - self.window if i >= self.window else 0
            high = i + self.window + 1 if i + self.window + 1 <= len(content_words) else len(content_words)
            window = content_words[low:i] + content_words[i+1:high]
            for co_word in window:
                hashed_feature = self._get_w_hash((co_word,))
                self.word_counts[WINDOW][hashed_w][hashed_feature] += 1

            sentence = content_words[0:i] + content_words[i+1:]
            for co_word in sentence:
                hashed_co_word = self._get_w_hash((co_word,))
                self.word_counts[SENTENCE][hashed_w][hashed_co_word] += 1

            self.word_frequency[hashed_w] += 1


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
                if self.stats.word_frequency[w] < self.stats.WORD_FREQUENCY:
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
            return self.stats.total_w[method][u] / self.stats.total[method]
        return self.stats.word_counts[method][u][att] / self.stats.total[method]

    def get_similarities(self, target, method):
        similarity_result = Counter()

        for att, u_att_ppmi in self.word_vecs[method][target].items():
            for v, v_att_ppmi in self.att_vecs[method][att].items():
                similarity_result[v] += (u_att_ppmi * v_att_ppmi)

        for v in similarity_result:
            similarity_result[v] /= (self.l2_norm[method][target] * self.l2_norm[method][v])
        del similarity_result[target]
        return similarity_result


if __name__ == '__main__':

    start_time_total = time.time()

    start_time = time.time()
    file_ = 'wikipedia.tinysample.trees.lemmatized'
    stats = WordsStats(window=2, word_freq=1, attributes_word_freq=1, attributes_limit=100).fit(file=file_)
    print(f'Finished fit stats {(time.time() - start_time):.3f} sec')

    file_ = 'counts_words.txt'
    with open(file_, 'w') as f:
        f.writelines([f"{w[0]} {w[1]}\n" for w in stats.word_frequency.most_common(n=50)])

    start_time = time.time()
    word_sim = WordSimilarities(stats=stats).fit()
    print(f'Finished fit Similarities {(time.time() - start_time):.3f} sec')

    print()
    # for word in target_words:
    #     start_time = time.time()
    #     sent_similarities = word_sim.get_similarities(target_word=word, method=stats.SENTENCE)
        # window_similarities = word_sim.get_similarities(target_word=word, method=stats.WINDOW)
        # dep_similarities = word_sim.get_similarities(target_word=word, method=stats.DEPENDENCY)
        # print(word)
        # for (sent_word, sent_sim), (win_word, win_sim), (dep_word, dep_sim) in zip(sent_similarities.most_common(20), window_similarities.most_common(20), dep_similarities.most_common(20)):
        #     print(f"{sent_word:<20} {sent_sim:.3f}\t{win_word:<20} {win_sim:.3f}\t{dep_word:<20} {dep_sim:.3f}")
        # print('*********')

    print(f'Finished time: {(time.time() - start_time_total):.3f} sec')
