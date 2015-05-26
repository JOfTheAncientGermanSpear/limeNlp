import json


class MonadLite:

    def __init__(self, dicts_list):
        self._values = dicts_list

    def values(self):
        return self._values

    def take(self, field):
        """
        >>> m = MonadLite([{'a': 3},{'a':4}])
        >>> m.take('a').values()
        [3, 4]
        """
        new_values = [d[field] for d in self.values() if field in d]
        return MonadLite(new_values)

    def flat_take(self, field):
        """
        >>> m = MonadLite([{'a': range(3)}, {'a':range(4,7)}, {'b': [10]}])
        >>> m.flat_take('a').values()
        [0, 1, 2, 4, 5, 6]
        """
        return MonadLite([v for d in self.values() for v in d.get(field, [])])

    def group_by(self, fn):
        """
        >>> m = MonadLite([{'a': range(3)}, {'a':range(4,7)}])
        >>> m.group_by(lambda acc, d: acc + d['a']).values()
        [0, 1, 2, 3, 4, 5, 6]
        """
        def agg(acc, d):
            return fn(acc, d)
        return MonadLite(reduce(agg, self.values(), dict()))

    def filter(self, fn):
        return MonadLite([d for d in self.values() if fn(d)])

    def map(self, fn):
        return MonadLite([fn(d) for d in self.values()])

    def __len__(self):
        return len(self.values())


def load_parsed(f_name):
    with open(f_name) as f:
        content = json.load(f)
    return content


def combine_sentences(f_names):
    def add_f(sents, f_name):
        js = load_parsed(f_name)
        curr_sents = js.get('sentences', [])
        return sents + curr_sents

    sentences = reduce(add_f, f_names, [])
    return MonadLite(sentences)