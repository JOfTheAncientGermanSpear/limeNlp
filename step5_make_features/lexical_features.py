from __future__ import division

import os

import pandas as pd

import lime_utils
import parse_utils


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

    sentences = parse_utils.MonadLite(sentences)

    def merge_lemmas(acc, token):
        lemma = token['lemma'].lower()
        if lemma in norms.index:
            pos = token['pos']
            acc[pos] = acc.get(pos, []) + [lemma]
        return acc

    tokens = sentences.flat_take('tokens')

    pos_lemmas = tokens.group_by(merge_lemmas).values()

    def get_mean(ls):
        mn = norms.loc[ls].mean()
        return [(norm_type, mn[norm_type]) for norm_type in mn.index]

    ret = {pos + '_' + norm_type: v
           for pos, ls in pos_lemmas.iteritems()
           for norm_type, v in get_mean(ls)}

    all_lemmas = tokens.take('lemma').\
        map(lambda l: l.lower()).\
        filter(lambda l: l in norms.index).\
        values()

    for norm_type, v in get_mean(all_lemmas):
        ret['overall_' + norm_type] = v

    return ret


def dirs_to_mat(patients_parse_dir="../data/patients/step3_parse_txt/",
                controls_parse_dir="../data/controls/step3_parse_txt/",
                bristol_csv="../data/norms/bristol_norms_30_08_05.csv",
                g_l_csv="../data/norms/gl_rate.csv"):
    merged_norms = merge_norms(bristol_csv, g_l_csv)

    def calc_norms(src_dir):
        def merge_sentences(acc, f_name):
            js = parse_utils.load_parsed(f_name)
            num = lime_utils.lime_num(f_name)
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
    patient_norms['had_stroke'] = 1
    control_norms['had_stroke'] = 0

    all_norms = pd.concat([patient_norms, control_norms])

    return all_norms


if __name__ == "__main__":
    import doctest
    doctest.testmod()
