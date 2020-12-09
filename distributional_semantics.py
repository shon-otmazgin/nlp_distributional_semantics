import time
from collections import defaultdict, Counter
from itertools import groupby
import math
from utils import *
from tqdm import tqdm
import pandas as pd
from memory_profiler import profile


class WordsStats:
    def __init__(self, window, attributes_word_freq, attributes_limit):
        self.window = window
        self.word_frequency = Counter()
        self.word_counts = defaultdict(lambda: defaultdict(Counter))   # word_counts[method][word][attribute] = #freq
        self.total = {SENTENCE: 0, WINDOW: 0, DEPENDENCY: 0}
        self.total_smooth = {SENTENCE: 0, WINDOW: 0, DEPENDENCY: 0}
        self.total_w = {SENTENCE: defaultdict(lambda: 0), WINDOW: defaultdict(lambda: 0), DEPENDENCY: defaultdict(lambda: 0)}
        self.total_w_smooth = {SENTENCE: defaultdict(lambda: 0), WINDOW: defaultdict(lambda: 0), DEPENDENCY: defaultdict(lambda: 0)}
        self.total_att = {SENTENCE: defaultdict(lambda: 0), WINDOW: defaultdict(lambda: 0), DEPENDENCY: defaultdict(lambda: 0)}
        self.int2str = {}
        self.attributes_word_freq = attributes_word_freq
        self.attributes_limit = attributes_limit

    def _get_hash(self, s):
        hashed_s = hash(s)
        if hashed_s in self.int2str:
            return hashed_s
        self.int2str[hashed_s] = s
        return hashed_s

    def _get_s(self, hashed_s):
        return self.int2str[hashed_s]

    @profile
    def fit(self, file):
        def tokenize(row):
            return {h: v for h, v in zip(FIELDS_H, row.rstrip('\n').split('\t'))}
        #
        # def read_sentences():
        #     with open(file, 'r', encoding='utf8') as f:
        #         sentence = []
        #         for row in f:
        #             if row != '\n':
        #                 sentence.append(tokenize(row))
        #             else:
        #                 yield sentence
        #                 sentence = []
        # for sent in read_sentences():
        #     continue

        with open(file, 'r', encoding='utf8') as f:
            sentence = []
            for row in f:
                if row == '\n':
                    df = pd.DataFrame([r.rstrip('\n').rsplit('\t') for r in sentence], columns=FIELDS_H)
                    sentence = []
                else:
                    sentence.append(row)

        # # print(f'Fittinga {len(sentences)} sentences')


        #     content_words, prep_words = WordsStats.get_content_and_prep_words(sent)
        #     self.words_co_occurring(content_words=content_words)
        #     self.words_dependency(sentence_tokenized=sent, content_words=content_words, prep_words=prep_words)
        #
        # self.filter_stats()
        #
        # for method in self.word_counts:
        #     for hashed_w, w_c in self.word_counts[method].items():
        #         # cache p(u,*)
        #         self.total_w[method][hashed_w] = sum(w_c.values())
        #         self.total_w_smooth[method][hashed_w] = sum([v ** 0.75 for v in w_c.values()])
        #
        #         # cache p(*,*)
        #         self.total[method] += self.total_w[method][hashed_w]
        #         self.total_smooth[method] += self.total_w_smooth[method][hashed_w]
        #
        #         # cache p(*,att)
        #         for hashed_att, att_c in w_c.items():
        #             self.total_att[method][hashed_att] += att_c
        # return self

    @staticmethod
    def is_content_word(w):
        # if w[POSTAG] == 'RB':
        #     if w[LEMMA] not in STOP_WORDS:
        #         print(f'not in stop words: {w[LEMMA]}')
        #     else:
        #         print(f'in stop words: {w[LEMMA]}')
        return w[POSTAG] in CONTENT_WORD_TAGS and w[LEMMA] not in STOP_WORDS

    @staticmethod
    def get_content_and_prep_words(sentence_tokenized):
        content_words, prep_words = [], []
        for w in sentence_tokenized:
            if WordsStats.is_content_word(w=w):
                content_words.append(w)
            if w[POSTAG] in PREPOSITION:
                prep_words.append(w)
        return content_words, prep_words

    def set_attribute(self, w, att, method):
        hashed_w = self._get_hash(s=w)
        hashed_att = self._get_hash(s=att)
        self.word_counts[method][hashed_w][hashed_att] += 1

    def words_dependency(self, sentence_tokenized, content_words, prep_words):
        def build_dependency_attribute(w):
            label, co_word = None, None
            parent_w = sentence_tokenized[int(w[HEAD]) - 1] if int(w[HEAD]) > 0 else None

            if parent_w and WordsStats.is_content_word(w=parent_w):
                label = w[DEPREL]
                co_word = parent_w[LEMMA]

            elif parent_w in prep_words:
                parent_parent_w = sentence_tokenized[int(parent_w[HEAD]) - 1] if int(parent_w[HEAD]) > 0 else None
                if parent_parent_w:
                    label = f'{parent_w[DEPREL]}:{parent_w[LEMMA]}'
                    co_word = parent_parent_w[LEMMA]

            if label and co_word:
                att_w = (co_word, label, IN)
                self.set_attribute(w=w[LEMMA], att=att_w, method=DEPENDENCY)

                att_co_w = (w[LEMMA], label, OUT)
                self.set_attribute(w=co_word, att=att_co_w, method=DEPENDENCY)

        for w in content_words:
            build_dependency_attribute(w)

    def filter_stats(self):
        """
        Filtering Features:
        attribute is a tuple of (word, att1(optional), att1(optional), ...)
        100 most_common features for a word
        frequent of the attribute's word (location 0 in the tuple) should be grater than 75
        """
        for method in self.word_counts:
            for hashed_w in self.word_counts[method]:
                c = Counter()
                for hashed_att, count in self.word_counts[method][hashed_w].most_common(n=self.attributes_limit):
                    att = self._get_s(hashed_att)
                    hashed_w_att = self._get_hash(att[0])
                    if self.word_frequency[hashed_w_att] >= self.attributes_word_freq:
                        c[hashed_att] = count
                self.word_counts[method][hashed_w] = c

    def words_co_occurring(self, content_words):
        for i, w in enumerate(content_words):
            low = i - self.window if i >= self.window else 0
            high = i + self.window + 1 if i + self.window + 1 <= len(content_words) else len(content_words)
            window = content_words[low:i] + content_words[i+1:high]
            for co_word in window:
                self.set_attribute(w=w[LEMMA], att=(co_word[LEMMA],), method=WINDOW)

            sentence = content_words[0:i] + content_words[i+1:]
            for co_word in sentence:
                self.set_attribute(w=w[LEMMA], att=(co_word[LEMMA],), method=SENTENCE)

            hashed_w = self._get_hash(s=w[LEMMA])
            self.word_frequency[hashed_w] += 1


class WordSimilarities:
    def __init__(self, word_freq, stats: WordsStats, smooth_ppmi=True):
        self.smooth_ppmi = smooth_ppmi
        self.stats = stats
        self.l2_norm = defaultdict(lambda: defaultdict(lambda: 0.))
        self.word_vecs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))) # for efficient cosine similarity
        self.att_vecs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))  # for efficient cosine similarity
        self.word_freq = word_freq

    def fit(self):
        for method in self.stats.word_counts:
            p_all, p_u, p_att = 0, 0, Counter()
            for w, counter in self.stats.word_counts[method].items():
                for att in counter:
                    if att not in p_att:
                        p_att[att] += self.p(att=att, method=method)
                    p_all += self.p(u=w, att=att, method=method)
                p_u += self.p(u=w, method=method)
            print(f'{method:<15} p1: {p_all:<20} p1: {p_u:<20} p1: {sum(p_att.values()):<20}')

        # pre process - create ppmi vector to each word and also calc the norm.
        for method in self.stats.word_counts:
            for w in self.stats.word_counts[method]:
                if self.stats.word_frequency[w] < self.word_freq:
                    continue
                for att in self.stats.word_counts[method][w]:
                    ppmi = self.get_PPMI(u=w, att=att, method=method)
                    self.l2_norm[method][w] += ppmi**2
                    self.word_vecs[method][w][att] = ppmi
                    self.att_vecs[method][att][w] = ppmi

                self.l2_norm[method][w] = math.sqrt(self.l2_norm[method][w])
        return self

    def get_PPMI(self, u, att, method):
        ppmi = math.log(
            self.p(u=u, att=att, method=method) / (self.p(u=u, method=method) * self.p(att=att, method=method)))

        return max(ppmi, 0)

    def p(self, method, u=None, att=None):
        if u is None and att is None:
            return 1
        if u is None:
            return self.stats.total_att[method][att] / self.stats.total[method]
        if att is None:
            if self.smooth_ppmi:
                return self.stats.total_w_smooth[method][u] / self.stats.total_smooth[method]
            else:
                return self.stats.total_w[method][u] / self.stats.total[method]
        return self.stats.word_counts[method][u][att] / self.stats.total[method]

    def get_cosine_similarities(self, target, method):
        hashed_target = hash(target)
        hashed_similarity_result = Counter()
        similarity_result = Counter()

        for att, u_att_ppmi in self.word_vecs[method][hashed_target].items():
            for v, v_att_ppmi in self.att_vecs[method][att].items():
                hashed_similarity_result[v] += (u_att_ppmi * v_att_ppmi)

        del hashed_similarity_result[hashed_target]

        for v, sim in hashed_similarity_result.items():
            similarity_result[self.stats.int2str[v]] = sim / (self.l2_norm[method][hashed_target] * self.l2_norm[method][v])
        return similarity_result


if __name__ == '__main__':

    start_time_total = time.time()

    start_time = time.time()
    file = 'wikipedia.tinysample.trees.lemmatized'

    stats = WordsStats(window=2, attributes_word_freq=75, attributes_limit=100).fit(file=file)
    print(f'Finished fit stats {(time.time() - start_time):.3f} sec')

    # file_ = 'counts_words.txt'
    # with open(file_, 'w') as f:
    #     f.writelines([f"{stats.int2str[w]} {count}\n" for w, count in stats.word_frequency.most_common(n=50)])
    #
    # file_ = 'counts_contexts_dep.txt'
    # with open(file_, 'w') as f:
    #     dep_context = Counter()
    #     for w, c in stats.word_counts[DEPENDENCY].items():
    #         dep_context += c
    #     f.writelines([f"{stats.int2str[dep]} {count}\n" for dep, count in dep_context.most_common(n=50)])

    # start_time = time.time()
    # word_sim = WordSimilarities(word_freq=100, stats=stats, smooth_ppmi=False).fit()
    # print(f'Finished fit Similarities {(time.time() - start_time):.3f} sec')
    #
    # file_ = 'top20.txt'
    # with open(file_, 'w') as f:
    #     for word in target_words:
    #         sent_sim = word_sim.get_cosine_similarities(target=word, method=SENTENCE).most_common(20)
    #         win_sim = word_sim.get_cosine_similarities(target=word, method=WINDOW).most_common(20)
    #         dep_sim = word_sim.get_cosine_similarities(target=word, method=DEPENDENCY).most_common(20)
    #         print(word)
    #         print(len(sent_sim), len(win_sim), len(dep_sim))
    #
    #         f.write(f'{word}\n')
    #         for (sent_w, sent_s), (win_w, win_s), (dep_w, dep_s) in zip(sent_sim, win_sim,  dep_sim):
    #             print(f"{win_w:<20} {win_s:.3f}\t{sent_w:<20} {sent_s:.3f}\t{dep_w:<20} {dep_s:.3f}")
    #
    #             f.write(f"{win_w:<20} {sent_w:<20} {dep_w:<20}\n")
    #
    #         print(f'*********')
    #         f.write(f'*********\n')
    #
    # print(f'Finished time: {(time.time() - start_time_total):.3f} sec')
    # print('Not Smoothed!')