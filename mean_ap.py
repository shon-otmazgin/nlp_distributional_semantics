from collections import defaultdict

from utils import WINDOW, SENTENCE, DEPENDENCY

TOPIC = 'topic'
SEMANTIC = 'semantic'

relevant_words = defaultdict(lambda: defaultdict(lambda: defaultdict(bool)))
retrieved = defaultdict(lambda: defaultdict(list))

input_file = 'annotation_output'
with open(input_file, 'r', encoding='utf8') as f:
    for row in f:
        values = row.rstrip('\n').split()
        if len(values) == 1:
            t_word = values[0]
        elif values:
            for i in range(0, len(values), 3):
                w, t_a, s_a = values[i:i+3]
                relevant_words[t_word][w][TOPIC] = True if t_a == '+' else False
                relevant_words[t_word][w][SEMANTIC] = True if s_a == '+' else False

            retrieved[t_word][WINDOW].append(values[0])
            retrieved[t_word][SENTENCE].append(values[3])
            retrieved[t_word][DEPENDENCY].append(values[6])


print(relevant_words['car'].keys())
print(len(relevant_words['car'].keys()))
print(relevant_words['piano'].keys())
print(len(relevant_words['piano'].keys()))
print(retrieved['car'][WINDOW])
print(retrieved['car'][SENTENCE])
print(retrieved['car'][DEPENDENCY])
