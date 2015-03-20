import json
import sys

import matplotlib
import web
        
sys.path.insert(0, '../step3_parse_txt/')
sys.path.insert(0, '../step4_make_features/')
import text_parser
import tree_generator

matplotlib.use('Agg')

urls = (
    '/(.*)', 'file_maker'
)
app = web.application(urls, globals())

class file_maker:        
    def GET(self, text):
        parse_res = text_parser.parse(text)
        cpd = tree_generator.gen_from_json(parse_res)
        files = ['corefs', 'phrase', 'deps']
        for i in range(2):
            tree_generator.print_tree(cpd[i], 'static/{}.ps'.format(files[i]))
        return "<a href='static/corefs.ps'></a>"

if __name__ == "__main__":
    app.run()
