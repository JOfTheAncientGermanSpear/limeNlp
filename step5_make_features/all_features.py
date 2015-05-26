from __future__ import division

import numpy as np
import pandas as pd

import norms_features
import freq_features
import phrase_features


def sanitize_feature_matrix(mat):
    """
    >>> import pandas as pd
    >>> import numpy as np
    >>> rand10 = lambda: np.random.randn(10)
    >>> no_var = ('no_var', np.ones(10))
    >>> high_nan_vals = rand10()
    >>> high_nan_vals[::5] = np.nan
    >>> high_nan = ('high_nan', high_nan_vals)
    >>> punct = (',_dist', rand10())
    >>> comma_count = (',_count', rand10())
    >>> count_vals = np.ones(10)
    >>> count_vals[::2] = np.nan
    >>> count = ('x_count', count_vals)
    >>> ratio_vals = np.ones(10)
    >>> ratio_vals[::2] = np.nan
    >>> ratio = ('x_ratio', ratio_vals)
    >>> ok = ('x_y_avg_dist', rand10())
    >>> mat = pd.DataFrame(dict([no_var, high_nan, punct, comma_count, count, ratio, ok]))
    >>> mat_f = sanitize_feature_matrix(mat)
    >>> sorted(mat_f.columns)
    ['x_count', 'x_ratio', 'x_y_avg_dist']
    >>> mat_f['x_count'].values
    array([ 0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.])
    >>> mat_f['x_ratio'].values
    array([ 0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.])
    """
    (num_samples, num_features) = mat.shape

    puncts = [',', '.', ';', ':']

    count_based = lambda c: '_count' in c or '_ratio' in c
    high_nan = lambda c: sum(np.isnan(mat[c])) > num_samples * .2
    has_punct = lambda c: any([p in c for p in puncts])
    no_var = lambda c: np.var(mat[c]) == 0
    drop = lambda c: has_punct(c) or high_nan(c) or no_var(c)

    mat_f = mat
    for l in filter(count_based, mat.columns):
        mat_f.loc[np.isnan(mat[l]), l] = 0

    mat_f = mat_f.drop(filter(drop, mat.columns), axis=1)

    return mat_f


def load():
    print("about to calculate phrase")
    phrase_controls_dir = '../data/controls/step4_make_trees'
    phrase_patients_dir = '../data/patients/step4_make_trees'
    phrase = phrase_features.\
        dirs_to_csv(phrase_patients_dir, phrase_controls_dir)
    phrase = sanitize_feature_matrix(phrase)

    print("about to calculate norms")
    parsed_controls_dir = '../data/controls/step3_parse_txt/'
    parsed_patients_dir = '../data/patients/step3_parse_txt/'
    bristol_csv = '../data/norms/bristol_norms_30_08_05.csv'
    g_l_csv = '../data/norms/gl_rate.csv'
    norms = norms_features.\
        dirs_to_csv(parsed_patients_dir, parsed_controls_dir, bristol_csv, g_l_csv)

    print("about to calculate frequency")
    freqs = freq_features.\
        dirs_to_csv(parsed_patients_dir, parsed_controls_dir)

    def prep_lu_data(df):
        df.index = [phrase_features.lime_num(id) for id in df.id]
        del df['id']

    print("about to calculate lexical complexity")
    lexical_complexity = pd.read_csv('../data/step5/lu_lexical.csv')
    prep_lu_data(lexical_complexity)

    print("about to calculate syntax complexity")
    syntax_complexity = pd.read_csv('../data/step5/lu_syntax.csv')
    prep_lu_data(syntax_complexity)

    lexical = pd.concat([lexical_complexity, norms, freqs], axis=1)
    syntax = pd.concat([syntax_complexity, phrase], axis=1)

    return lexical, syntax


if __name__ == "__main__":
    import doctest
    doctest.testmod()