from collections import defaultdict

input_file = 'annotation_output'

topic_annotation = defaultdict(lambda: defaultdict(bool))
semantic_annotation = defaultdict(lambda: defaultdict(bool))

relevant_words = defaultdict(set)
retrived_words = defaultdict(lambda: defaultdict())
with open(input_file, 'r', encoding='utf8') as f:
    for row in f:
        values = row.rstrip('\n').split()
        if len(values) == 1:
            t_word = values[0]
        else:
            relevant_words[t_word].update()
            for i in range(0, len(values), 3):
                w, t_a, s_a = values[i:i+3]
                topic_annotation[t_word][w] = True if t_a == '+' else False
                semantic_annotation[t_word][w] = True if s_a == '+' else False



N = len(topic_annotation['car'].values())
print(N)
