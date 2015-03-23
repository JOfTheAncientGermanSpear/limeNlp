import hashlib
import json
import os
import re
from subprocess import call
import sys


import web
        
import sentence_form
sys.path.insert(0, '../step3_parse_txt/')
sys.path.insert(0, '../step4_make_features/')
import text_parser
import tree_generator

render = web.template.render("templates")

urls = (
    '/(.*)', 'file_maker'
)
app = web.application(urls, globals())

add_space = re.compile(r'([^ ])\]')
word_index = re.compile(r'word: ([^,]*), index: ')
colon_num = re.compile(r': ([0-9])+')
def sanitize(latex_text):
    """
    >>> sanitize('a]')
    'a ]'
    >>> sanitize('.word: big, index: 4')
    '.big_4'
    """
    text = add_space.sub(r'\1 ]', latex_text)
    text = word_index.sub(r'\1_', text)
    text = colon_num.sub(r'_\1', text)
    return text


def latex_wrap(tree):
    return """\\documentclass{book}
    \\usepackage{qtree}
    \\usepackage{adjustbox}
    \\begin{document}
    \\begin{adjustbox}{width=1\\textwidth}

    """ + sanitize(tree) + """\n
    \\end{adjustbox}
    \\end{document}"""

def parse_to_files(text):
        parse_res = text_parser.parse(text)
        cpd = tree_generator.gen_from_json(parse_res)
        files = ['corefs', 'phrase', 'deps']
        links = []
        res_dir = 'static/{}'.format(hashlib.sha1(text).hexdigest())
        os.mkdir(res_dir)
        for i in range(3):
            f_name = '{}/{}.TEX'.format(res_dir, files[i])
            with open(f_name, 'w') as f:
                f.write(latex_wrap(cpd[i].pprint_latex_qtree()))

            os.system("pdflatex " + f_name)
            os.system("mv *.aux *.pdf *.log {}".format(res_dir))

            pdf = '{}/{}.pdf'.format(res_dir, files[i])
            l = "<a href='" + pdf + "'>" + files[i] + "</a>"
            links.append(l)
        return links

class file_maker:        
    def GET(self, x):
        f = sentence_form.the_form()
        return render.home(f)

    def POST(self, x):
        f = sentence_form.the_form()
        f.validates()
        links = parse_to_files(f.d.text)
        return "<br/>".join(links)

def test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    app.run()
