from __future__ import division

import os
import re
import zipfile

import feature_generator


def _cat_pat(src_dir, lime_num, dst_file):
    files = [os.path.join(src_dir, f) for f in os.listdir(src_dir)
             if str(lime_num) in f and f.endswith('.txt')]
    with open(dst_file, 'w') as dst:
        def write(src_file):
            with open(src_file, 'r') as src:
                dst.write(src.read())

        [write(f) for f in files]


"http://pymotw.com/2/zipfile/"
def _create_empty_zip(f_name):

    try:
        import zlib
        compression = zipfile.ZIP_DEFLATED
    except:
        compression = zipfile.ZIP_STORED

    modes = {
        zipfile.ZIP_DEFLATED: 'deflated',
        zipfile.ZIP_STORED: 'stored'
    }

    return zipfile.ZipFile(f_name, mode='w'), modes[compression]


def _num_spaces(txt):
    """
    >>> x = 'hello there how are you'
    >>> _num_spaces(x)
    4
    """
    return len(re.findall(r'[^ ] [^ ]', txt))


def cat_pats(src_dir, dst_dir):
    lime_nums = {feature_generator.lime_num(f) for f in os.listdir(src_dir)
                 if feature_generator.lime_num(f)}

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


if __name__ == "__main__":
    import doctest
    doctest.testmod()