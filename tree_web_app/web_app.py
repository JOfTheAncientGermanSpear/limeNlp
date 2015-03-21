import json
import os
import re
from subprocess import call
import sys


import web
        
sys.path.insert(0, '../step3_parse_txt/')
sys.path.insert(0, '../step4_make_features/')
import text_parser
import tree_generator


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

class file_maker:        
    def GET(self, text):
        if not text:
            return """
<p>please enter a sentence in the URL</p>
<br/>
<p>For example:</p>
<a href="./The world is round and it moves.">The world is round and it moves</a>
        """
        parse_res = text_parser.parse(text)
        cpd = tree_generator.gen_from_json(parse_res)
        files = ['corefs', 'phrase', 'deps']
        links = []
        for i in range(3):
            f_name = 'static/{}.TEX'.format(files[i])
            with open(f_name, 'w') as f:
                f.write(latex_wrap(cpd[i].pprint_latex_qtree()))
            pdf = 'static/{}.pdf'.format(files[i])
            os.system("pdflatex " + f_name)
            os.system("mv *.aux *.pdf *.log static/")
            l = "<a href='" + pdf + "'>" + files[i] + "</a>"
            links.append(l)
        return "<br/>".join(links)

def test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    app.run()
