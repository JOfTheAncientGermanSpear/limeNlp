from __future__ import division

import os
import re

import pandas as pd

import tree_utils

def lime_num(s):
    """
    >>> lime_num('def/aoub1234_ax.txt')
    '1234'
    >>> lime_num('eu12/eu567_eu.txt')
    '567'
    >>> lime_num('eu12/eueu.txt')

    """
    res = re.findall(r'[0-9]{3,}', s)
    return res[0] if len(res) > 0 else None


def is_sentence_label(l):
    """
    >>> is_sentence_label('id: 0')
    True
    >>> is_sentence_label('id: 37')
    True
    >>> is_sentence_label('sentences')
    True
    >>> is_sentence_label('ab')
    False
    """
    l_lower = l.lower()
    return l_lower == "sentences" or 'id: ' in l_lower


def lime_nums_to_mat(patients_dir, controls_dir, src_filter, row_fn):

    def calc_features(src_dir):
        def add_tree_to_map(d, f):
            lm = lime_num(f)
            if lm is not None:
                full_f = os.path.join(src_dir, f)
                t = tree_utils.load_tree(full_f)
                curr = d.get(lm, [])
                curr.append(t)
                d[lm] = curr
            return d

        id_trees_map = reduce(add_tree_to_map, filter(src_filter, os.listdir(src_dir)), dict())

        merge_sentence_sub_nodes = lambda ts: tree_utils.merge_trees(ts, 'sentences', replace_roots=True)

        id_tree_mappings = map(lambda i_ts: (i_ts[0], merge_sentence_sub_nodes(i_ts[1])), id_trees_map.items())

        id_rows_map = dict(map(lambda it: (it[0], row_fn(it[1])), id_tree_mappings))

        mat = pd.DataFrame(id_rows_map)
        mat = mat.T
        return mat

    patient_features = calc_features(patients_dir)
    control_features = calc_features(controls_dir)
    patient_features['has_aphasia'] = 1
    control_features['has_aphasia'] = 0

    all_features = pd.concat([patient_features, control_features])

    return all_features


if __name__ == "__main__":
    import doctest
    doctest.testmod()
