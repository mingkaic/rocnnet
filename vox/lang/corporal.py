import random
import json

class cGenerator:
    def __init__(self, ss, tl):
        self.top_level = tl
        self.struct = ss
        assert(tl in ss)

    def gen_clause(self):
        clause = random.choice(self.struct[self.top_level])
        while True:
            no_subclause = True
            next_clause = []
            for word in clause:
                if word in self.struct:
                    no_subclause = False
                    next_clause += random.choice(self.struct[word])
                else:
                    next_clause.append(word)
            if no_subclause:
                break
            clause = next_clause
        return clause

def make_corpus(gfile, vfile, minlin, nline):
    gram_file = open(gfile, 'r')
    g = json.loads(gram_file.read())
    cmap = []
    cs = g.get("class", {})
    for c in cs:
        cmap += [(word, c) for word in cs[c]]
    cmap = dict(cmap)
    vocab = dict()
    with open(vfile, 'r') as f:
        vocab = dict([(word, cmap.get(word, 'noun'))
            for word in f.read().lower().split()])
    classes = dict()
    for word in vocab:
        cl = vocab[word]
        if cl in classes:
            classes[cl].append(word)
        else:
            classes[cl] = [word]

    structure = g.get("structure", {})
    for substruct in structure:
        sentences = structure[substruct]
        structure[substruct] = [
            sentence
            for sentence in sentences
                if all([
                    wclass in classes or wclass in structure
                    for wclass in sentence
                ])
        ]
    gen = cGenerator(structure, g.get("top_level"))
    order = list(vocab.keys())
    word2idx = dict([(word, i) for i, word in enumerate(order)])

    # repeatedly select clauses until the line is long enough
    lines = []
    for _ in range(nline):
        sentence = []
        while len(sentence) < minlin:
            sentence += gen.gen_clause()
        lines.append([
            word2idx[random.choice(classes[cl])]
            for cl in sentence
        ])
    return lines, order, vocab

def stringify_corpus(corpus, order):
    return [[order[word] for word in line] for line in corpus]

if '__main__' == __name__:
    corpus, order, vocab = make_corpus('grammar.json', 'vocabulary3.txt', 100, 1000)
    print(vocab)
    print(stringify_corpus(corpus, order))
