import json
import os
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

def latex_wrap(tree):
    return """\\documentclass{book}
    \\usepackage{qtree}
    \\begin{document}

    """ + tree + "\n\\end{document}"

class file_maker:        
    def GET(self, text):
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

if __name__ == "__main__":
    app.run()
