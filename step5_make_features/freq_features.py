from __future__ import division

import os

import json
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

import lime_utils
import parse_utils

stopwords_en = set(stopwords.words('english'))

def subtlex_freqs():
    f_name = "../data/subtlex_counts.json"

    with open(f_name) as f:
        counts = json.load(f)

    def merge_into_lower(word_to_merge):
        word_lower = word_to_merge.lower()
        counts[word_lower] = counts.get(word_lower, 0) + counts[word_to_merge]

    counts_upper_words = {w for w in counts if w.istitle() or w.isupper()}

    for word in counts.keys():
        if word in counts_upper_words:
            merge_into_lower(word)

    counts_stopwords = {w for w in counts if w in stopwords_en}

    for w in counts_upper_words.union(counts_stopwords):
        del counts[w]

    num_words = sum(counts.itervalues())

    return pd.Series({w: -1 * np.log10(c / num_words) for w, c in counts.iteritems()})


_subtlex_freqs = subtlex_freqs()


def average_freq(f_names):
    sentences = parse_utils.combine_sentences(f_names)

    def merge_lemmas(acc, token):
        lemma = token['lemma'].lower()
        if lemma in _subtlex_freqs:
            pos = token['pos']
            acc[pos] = acc.get(pos, []) + [lemma]
        return acc

    tokens = sentences.flat_take('tokens')

    pos_lemmas = tokens.group_by(merge_lemmas).values()

    ret = {p + '_subtlex_freq': _subtlex_freqs[ls].mean() for p, ls in pos_lemmas.iteritems()}

    subtlex_lemmas = tokens.take('lemma').\
        map(lambda l: l.lower()).\
        filter(lambda l: l in _subtlex_freqs).values()

    ret['overall_subtlex_freq'] = _subtlex_freqs[subtlex_lemmas].mean()

    alpha_num_lemmas = tokens.take('lemma').filter(lambda l: l.isalnum())

    ret['word_count'] = len(alpha_num_lemmas)

    ret['avg_letter_count'] = np.mean([len(l) for l in alpha_num_lemmas.values()])

    return ret

fillers = {'ugh', 'um', 'uh', 'ah', 'er'}

def filler_count(f_names):
    def add_text(txt, f_name):
        with open(f_name, 'r') as f:
            curr_txt = f.read().decode('utf-8')
        return txt + curr_txt

    content = reduce(add_text, f_names, '')

    words = word_tokenize(content)
    return len([w for w in words if w.lower() in fillers])


def dirs_to_mat(patients_parsed_dir, controls_parsed_dir, patients_txt_dir, controls_txt_dir):
    def merge_fs(acc, f):
        l_num = lime_utils.lime_num(f)
        acc[l_num] = acc.get(l_num, []) + [f]
        return acc

    def fs_by_lime_num(src_dir):
        full_fs = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]
        return reduce(merge_fs, full_fs, dict())

    def get_freqs(src_dir):
        fs_avg = {n: average_freq(fs) for n, fs in fs_by_lime_num(src_dir).iteritems()}
        return pd.DataFrame(fs_avg).T

    patients = get_freqs(patients_parsed_dir)
    controls = get_freqs(controls_parsed_dir)

    patients['had_stroke'] = 1
    controls['had_stroke'] = 0

    def get_fillers(src_dir):
        is_txt = lambda f: f.endswith('txt')
        filler_counts = {n: filler_count([f for f in fs if is_txt(f)])
                         for n, fs in
                         fs_by_lime_num(src_dir).iteritems()}
        return pd.Series(filler_counts)

    patient_fillers = get_fillers(patients_txt_dir)
    control_fillers = get_fillers(controls_txt_dir)

    patients['filler_count'] = patient_fillers
    controls['filler_count'] = control_fillers

    full = pd.concat([patients, controls])

    return full
