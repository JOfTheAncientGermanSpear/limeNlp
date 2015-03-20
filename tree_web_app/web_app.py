import json
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

class file_maker:        
    def GET(self, text):
        parse_res = text_parser.parse(text)
        cpd = tree_generator.gen_from_json(parse_res)
        files = ['corefs', 'phrase', 'deps']
        links = []
        for i in range(3):
            f = 'static/{}.ps'.format(files[i])
            tree_generator.print_tree(cpd[i], f)
            l = "<a href='" + f + "'>" + files[i] + "</a>"
            links.append(l)
        return "<br/>".join(links)

if __name__ == "__main__":
    app.run()
