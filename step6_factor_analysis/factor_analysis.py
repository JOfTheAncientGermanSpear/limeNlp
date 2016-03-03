import numpy as np
import pandas as pd


def _data_path(f):
    return "../data/step5/%s.csv" % f


def _remove_aphasia_dupe_cols(df):
    """
    :param df:
    :return:
    >>> import pandas as pd
    >>> df = pd.DataFrame(dict(has_aphasia=[1, 2], has_aphasia_1=[1, 2]))
    >>> _remove_aphasia_dupe_cols(df)
    >>> assert set(df.columns) == {'has_aphasia'}
    >>> assert df.shape == (2, 1)
    >>> df_diff = pd.DataFrame(dict(has_aphasia=[1, 2], has_aphasia_1=[4, 3]))
    >>> _remove_aphasia_dupe_cols(df_diff)
    Traceback (most recent call last):
    ...
    AssertionError
    """
    for c in df.columns:
        if c.startswith('has_aphasia') and c != 'has_aphasia':
            assert np.all(df['has_aphasia'] == df[c])
            del df[c]


def load(dep_path=_data_path('dependencies'),
         lex_path=_data_path('lexical'),
         syn_path=_data_path('syntax')):

    def rd(f):
        ret = pd.read_csv(f, index_col=0)

        valid_rows = np.logical_not(np.isnan(ret['has_aphasia']))
        ret = ret.loc[valid_rows, :]

        _remove_aphasia_dupe_cols(ret)

        return ret

    return rd(dep_path), rd(lex_path), rd(syn_path)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
