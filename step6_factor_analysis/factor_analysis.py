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


def _remove_na_cols(df, thresh=.7):
    """
    :param df:
    :param thresh:
    :param copy:
    :return:
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(dict(one_four=[1, 2, 3, np.nan], pure=[1, 2, 3, 4]))
    >>> assert 'one_four' not in _remove_na_cols(df, thresh=.76)
    >>> assert 'pure' in _remove_na_cols(df, thresh=.76)
    >>> assert 'one_four' in _remove_na_cols(df)
    >>> assert 'pure' not in _remove_na_cols(df, thresh=1.0)
    """

    num_rows = df.shape[0]
    max_nans = (1 - thresh) * num_rows

    cols_to_keep = [c for c in df.columns if df[c].isnull().sum() < max_nans]
    return df[cols_to_keep]


def _remove_na_rows(df, thresh=.7):
    """
    :param df:
    :param thresh:
    :param copy:
    :return:
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(dict(a=[1, 2, 3, np.nan], b=[1, 2, 3, 4], c=[2, 4, np.nan, np.nan]))
    >>> assert _remove_na_rows(df).shape == (2, 3)
    >>> assert _remove_na_rows(df, thresh=.6).shape == (3, 3)
    """

    num_cols = df.shape[1]
    max_nans = (1 - thresh) * num_cols

    rows_to_keep = [np.isnan(vals).sum() < max_nans for _, vals in df.iterrows()]
    return df.loc[rows_to_keep, :]


def _fillna(df, val_fn=lambda df: df.mean()):
    """
    :param df:
    :return:
    >>> import pandas as pd
    >>> df = pd.DataFrame(dict(a=[1, np.nan, 3], b=[10, np.nan, 30]))
    >>> df = _fillna(df)
    >>> assert np.all(df['a'].get_values() == np.array([1, 2, 3]))
    >>> assert np.all(df['b'].get_values() == np.array([10.0, 20.0, 30.0]))
    """
    return df.fillna(val_fn(df))


def load_continuous(dep_path=_data_path('dependencies'),
                    lex_path=_data_path('lexical'),
                    syn_path=_data_path('syntax'),
                    col_thresh=None,
                    row_thresh=None,
                    fill_na_fn=lambda df: df.mean()):

    if col_thresh is None:
        col_thresh = dict(dep=.7, lex=.7, syn=.7)
    if row_thresh is None:
        row_thresh = dict(dep=.7, lex=.7, syn=.7)

    def rd(f, (c_thresh, r_thresh)):
        ret = pd.read_csv(f, index_col=0)

        valid_rows = np.logical_not(np.isnan(ret['has_aphasia']))
        ret = ret.loc[valid_rows, :]

        _remove_aphasia_dupe_cols(ret)

        ret = _remove_na_cols(ret, c_thresh)
        ret = _remove_na_rows(ret, r_thresh)
        ret = _fillna(ret, fill_na_fn)

        return ret

    def threshes(key):
        return col_thresh[key], row_thresh[key]

    return rd(dep_path, threshes('dep')), rd(lex_path, threshes('lex')), rd(syn_path, threshes('syn'))


def plot(df, x_col, y_col):
    colors = np.zeros((len(df), 3))
    has_aphasia_c = [.6, .8, .4]
    for i in range(len(df)):
        if df.iloc[i, :]['has_aphasia']:
            colors[i, :] = has_aphasia_c

    import matplotlib.pyplot as plt

    plt.scatter(x=df[x_col], y=df[y_col], c=colors)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("%s vs %s" % (y_col, x_col))
    plt.legend(['has_aphasia'])
    plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
