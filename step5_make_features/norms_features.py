from __future__ import division

import json
import os

from nltk.corpus import stopwords
import pandas as pd

import phrase_feature_generator as pfg


class MonadLite():

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

    def filter(self, fn):
        return MonadLite([d for d in self.values() if fn(d)])


def subtlex_counts(f_name="../data/subtlex_counts.json"):
    with open(f_name) as f:
        counts = json.load(f)

    def merge_into_lower(word_to_merge):
        word_lower = word_to_merge.lower()
        counts[word_lower] = counts.get(word_lower, 0) + counts[word_to_merge]

    stopwords_en = set(stopwords.words('english'))

    upper_words = {w for w in counts if w.istitle() or w.isupper()}

    for word in counts:
        if word in upper_words:
            merge_into_lower(word)

    stop_words = {w for w in counts if w in stopwords_en}

    for w in upper_words.union(stop_words):
        del counts[w]

    return counts


def merge_norms(bristol_csv, g_l_csv):
    bristol = pd.read_csv(bristol_csv)
    g_l = pd.read_csv(g_l_csv)

    bristol.AoA /= 100
    bristol.IMG /= 100
    bristol.FAM /= 100

    g_l.WORD = g_l.WORD.apply(lambda w: w.lower())

    merged = pd.concat([bristol, g_l])
    merged.index = merged.WORD
    merged = merged[['AoA', 'IMG', 'FAM']]

    return merged.groupby(lambda ix: ix).mean()


def _norm_means(sentences, norms):
    lemmas = MonadLite(sentences).flat_take('tokens').\
        take('lemma').\
        filter(lambda l: l in norms.index).values()
    return dict(norms.loc[lemmas].mean())


def _load_json(f_name):
    with open(f_name) as f:
        content = json.load(f)
    return content


def dirs_to_csv(patients_parse_dir, controls_parse_dir, bristol_csv, g_l_csv, output_csv):
    merged_norms = merge_norms(bristol_csv, g_l_csv)

    def calc_norms(src_dir):
        def merge_sentences(acc, f_name):
            js = _load_json(f_name)
            num = pfg.lime_num(f_name)
            acc[num] = acc.get(num, [])
            acc[num] = acc[num] + js.get('sentences', [])
            return acc

        sentences_by_limenum = reduce(merge_sentences,
                                      [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith('json')],
                                      dict())
        norms_by_limenum_dict = {num: _norm_means(sentences, merged_norms)
                                 for (num, sentences) in sentences_by_limenum.iteritems()}
        norms_by_limenum_df = pd.DataFrame(norms_by_limenum_dict)
        return norms_by_limenum_df.T

    patient_norms = calc_norms(patients_parse_dir)
    control_norms = calc_norms(controls_parse_dir)
    patient_norms['has_aphasia'] = 1
    control_norms['has_aphasia'] = 0

    norms_series = pd.concat([patient_norms, control_norms])
    norms_series.to_csv(output_csv)
    return norms_series


if __name__ == "__main__":
    import doctest
    doctest.testmod()