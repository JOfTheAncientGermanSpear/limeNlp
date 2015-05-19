from __future__ import division

import pandas as pd


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