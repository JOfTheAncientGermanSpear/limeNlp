from __future__ import division

import os
import re
import zipfile

import pandas as pd

import phrase_feature_generator


def _cat_pat(src_dir, lime_num, dst_file):
    files = [os.path.join(src_dir, f) for f in os.listdir(src_dir)
             if str(lime_num) in f and f.endswith('.txt')]
    with open(dst_file, 'w') as dst:
        def write(src_file):
            with open(src_file, 'r') as src:
                dst.write(src.read())

        [write(f) for f in files]


def _num_spaces(txt):
    """
    >>> x = 'hello there how are you'
    >>> _num_spaces(x)
    4
    """
    return len(re.findall(r'[^ ] [^ ]', txt))


def cat_pats(src_dir, dst_dir):
    lime_nums = {phrase_feature_generator.lime_num(f) for f in os.listdir(src_dir)
                 if phrase_feature_generator.lime_num(f)}

    def write_lime_num(num):
        dst_file = os.path.join(dst_dir, 'concat_{0}.txt'.format(num))
        _cat_pat(src_dir, num, dst_file)
        return dst_file

    #http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
    chunks = lambda l, n: [l[x: x + n] for x in range(0, len(l), n)]

    file_groups = chunks(sorted(lime_nums), 30)

    for (i, lime_nums_per_group) in enumerate(file_groups):
        zf_name = os.path.join(dst_dir, '{0}.zip'.format(i))
        zf = zipfile.ZipFile(zf_name, mode='w')

        for num in lime_nums_per_group:
            dst_name = write_lime_num(num)
            with open(dst_name) as dst:
                content = dst.read()

            if _num_spaces(content) > 49:
                zf.write(dst_name)


def make_feature_matrices(pats_src_dir, cons_src_dir, dst_dir=None):
    is_csv_of_type = lambda f, t: t in f and f.endswith('.csv')

    type_files = lambda d, t: [os.path.join(d, f_name) for f_name in
                               os.listdir(d) if is_csv_of_type(f_name, t)]

    def add_file_to_df(df, f_name):
        d = pd.read_csv(f_name)
        return pd.concat([df, d])

    make_df_of_type = lambda d, t: \
        reduce(add_file_to_df, type_files(d, t), pd.DataFrame())

    pats_lex = make_df_of_type(pats_src_dir, 'lexical')
    pats_lex['has_aphasia'] = 1

    cons_lex = make_df_of_type(cons_src_dir, 'lexical')
    cons_lex['has_aphasia'] = 0

    pats_syn = make_df_of_type(pats_src_dir, 'syntax')
    pats_syn['has_aphasia'] = 1

    cons_syn = make_df_of_type(cons_src_dir, 'syntax')
    cons_syn['has_aphasia'] = 0

    lex = pd.concat([pats_lex, cons_lex])
    syn = pd.concat([pats_syn, cons_syn])

    if dst_dir is None:
        dst_dir = "."

    def write_csv(df, n):
        dst_csv = os.path.join(dst_dir, n + '.csv')
        df.to_csv(dst_csv, index=False)

    write_csv(lex, 'lexical')
    write_csv(syn, 'syntax')

    return lex, syn


if __name__ == "__main__":
    import doctest
    doctest.testmod()