import pandas as pd

import sys
sys.path.insert(0, '../utils/')
import utils


def _data_path(f):
    return "../data/step5/%s.csv" % f


def _remove_aphasia_dupe_cols(df):
    """
    :param df:
    :return:
    >>> import pandas as pd
    >>> x = pd.DataFrame
    """
    for c in df.columns:
        if c.startswith('has_aphasia') and c != 'has_aphasia':
            assert np.all(df['has_aphasia'] == df[c])
            del df[c]


def _load(dep_path=_data_path('dependencies'),
          lex_path=_data_path('lexical'),
          syn_path=_data_path('syntax')):

    def rd(f):
        ret = pd.read_csv(f, index_col=0)

        valid_rows = np.logical_not(np.isnan(ret['has_aphasia']))
        ret = ret.loc[valid_rows, :]

        _remove_aphasia_dupe_cols(ret)

        return ret

    return rd(dep_path), rd(lex_path), rd(syn_path)


def _create_index_from_id(df):
    """
    :param df:
    >>> import pandas as pd
    >>> df = pd.DataFrame(dict(a = [1, 2, 3], id = ['lm201', 'lm1234', 'lm205']))
    >>> _create_index_from_id(df)
    >>> expected = {201, 1234, 205}
    >>> actual = set(df.index)
    >>> assert actual == expected
    >>> assert len(df.index) == 3
    >>> assert 'id' not in df
    """
    df.index = [utils.lime_num(i + '.txt') for i in df['id']]
    del df['id']


def _load_lu(lex_path=_data_path('lu_lexical'),
             syn_path=_data_path('lu_syntax')):

    def rd(f):
        ret = pd.read_csv(f)
        _create_index_from_id(ret)
        return ret

    return rd(lex_path), rd(syn_path)


def load(deps_path=_data_path('dependencies'),
         lex_path=_data_path('lexical'),
         syn_path=_data_path('syntax'),
         lu_lex_path=_data_path('lu_lexical'),
         lu_syn_path=_data_path('lu_syntax')):

    deps, lex, syn = _load(deps_path, lex_path, syn_path)
    lu_lex, lu_syn = _load_lu(lu_lex_path, lu_syn_path)

    def concat(left, right):
        num_left_rows, num_left_cols = left.shape
        num_right_rows, num_right_cols = right.shape

        combined = pd.merge(left, right, left_index=True, right_index=True)
        import pdb
        pdb.set_trace()
        _remove_aphasia_dupe_cols(combined)

        return combined

    lex_all = concat(lex, lu_lex)
    syn_all = concat(syn, lu_syn)

    return deps, lex_all, syn_all



if __name__ == "__main__":
    import doctest
    doctest.testmod()
