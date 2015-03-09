from os import listdir
from os.path import join, basename

import json
import re

scenarios = ['circus', 'cookie_theft', 'picnic']

class DestInfo:
	def __init__(self, content_abs_filename, meta_abs_filename):
		self.content_abs_filename = content_abs_filename
		self.meta_abs_filename = meta_abs_filename

	def __str__(self):
		return 'content_abs_filename: {}'.format(self.content_abs_filename) + '\n' + 'meta_abs_filename: {}'.format(self.meta_abs_filename)

	def _write_to_file(self, abs_name, content):
		with open(abs_name, 'w') as f:
			f.write(content)

	def write(self, content = None, meta = None):
		if not (content or meta):
			raise Exception('at least content or meta must be set')

		if content:
			self._write_to_file(self.content_abs_filename, content)

		if meta:
			self._write_to_file(self.meta_abs_filename, json.dumps(meta))
	


def mk_title_re(scenario):
	return re.compile(r'(?:\n|\A).{,10}(' + scenario + r').{,20}(?:\n|\Z)', re.IGNORECASE)

title_res = {}
title_res['circus'] = mk_title_re('circus')
title_res['cookie_theft'] = mk_title_re('cooki?e ?theft')
title_res['picnic'] = mk_title_re('picnic')

def get_title_location(scenario, content):
	"""Get the title component of file content
	>>> get_title_location('circus', 'LM201PDcircus (0:15)')
	(0, 20)
	>>> get_title_location('picnic', 'hello there\\nLM201PDpicnic (0:15)\\n')
	(11, 33)
	>>> get_title_location('cookie_theft', '\\nCooke Theft')
	(0, 12)
	>>> get_title_location('cookie_theft', 'LM2013PDcookietheft (0:59 - 1:44)')
	(0, 33)
	"""
	ret = list(title_res[scenario].finditer(content))
	return (ret[0].start(), ret[0].end())

def get_scenario_content(scenario, scenarios, content):
	"""Return the content for a given scenario
	>>> get_scenario_content('circus', ['circus', 'cookie_theft'],'Circus\\nok clan um one\\nCooke Theft')
	'ok clan um one'
	>>> get_scenario_content('picnic', ['circus', 'cookie_theft', 'picnic'],'Circus\\nok clan um one\\nCooke Theft\\ncookie stuff, picnic over there\\nPicnic\\nThis is a picnic')
	'This is a picnic'
	"""
	scenario_ix = scenarios.index(scenario) 
	next_scenario = scenarios[scenario_ix + 1] if scenario_ix < len(scenarios) - 1  else None 
	
	(_, sce_end) = get_title_location(scenario, content)	
	(n_sce_start, _) = get_title_location(next_scenario, content) if next_scenario else (None, None)

	return content[sce_end:n_sce_start] if n_sce_start else content[sce_end:]


def _get_original_txt_files(path, file_filter_fn):
	f_names = [f for f in listdir(path) if file_filter_fn(f)]
	return [join(path, f) for f in f_names]


def _create_src_dest_map(src_paths, dest_path):
	ret = dict() 
	for s in src_paths:
		src_filename = basename(s)
		dest_filename = src_filename.replace(' ', '_')
		dest_filename = dest_filename.lower()
		dest_filename = join(dest_path, dest_filename)

		ret_s = dict() 
		for k in scenarios:
			dest_content_filename = dest_filename.replace('.txt', '_{}.txt'.format(k))
			dest_meta_filename = dest_filename.replace('.txt', '_' + k + '_convert_meta.json')
			ret_s[k] = DestInfo(dest_content_filename, dest_meta_filename)

		ret[s] = ret_s
	
	return ret

def _filter_content_lines(content, fn):
	lines = content.split('\n')
	filtered_lines = filter(fn, lines)
	num_removed = len(lines) - len(filtered_lines)
	return ('\n'.join(filtered_lines), num_removed)

comment_re = re.compile(r'\W*\(.*\)\W*')

def _remove_comments(content):
	return _filter_content_lines(content, lambda l: not comment_re.match(l))

unknown_re = re.compile(r'\W?\?')

def _remove_unknown_words(content):
	num_unknown = len(re.findall(unknown_re, content))
	sanitized_content = re.sub(unknown_re, '', content)
	return (sanitized_content, num_unknown)

def _remove_list_items(content):
	return _filter_content_lines(content, lambda l: len(l.split()) > 3)

def _sanitize_content(content):
	meta = dict()
	content, meta['unknown_words_count'] = _remove_unknown_words(content)
	content, meta['list_items_count'] = _remove_list_items(content)
	content, meta['comments_count'] = _remove_comments(content)
	return (content, meta)

def _create_prepped_txt_file(src, scenario_destinfo_map):
	with open(src) as src_f:
		content = src_f.read()

	def get_scenario_start(scenario):
		(start, _) = get_title_location(scenario, content)
		return start
	
	sorted_scenarios = sorted(scenarios, lambda l, r: get_scenario_start(l) - get_scenario_start(r))

        for scenario in sorted_scenarios:
		scenario_content = get_scenario_content(scenario, sorted_scenarios, content)
		scenario_content, scenario_meta = _sanitize_content(scenario_content)

		destinfo = scenario_destinfo_map[scenario]
		destinfo.write(content = scenario_content, meta = scenario_meta)


num_re = re.compile('[^0-9]*([0-9]+)[^0-9]*')
def lime_num(filename):
	""" Extract number out of a string
	>>> lime_num('LM201PD.txt')
	201
	>>> lime_num('LIME 1001 Picture Description Original.txt')
	1001
	"""
	return int(num_re.findall(filename)[0])


def create_prepped_txt_files(src_path, dest_path, start_from = 1, txt_filter= lambda f: f.endswith('Original.txt')):

	srcs = _get_original_txt_files(src_path, txt_filter)
	srcs = [s for s in srcs if lime_num(basename(s)) > start_from]

	src_dest_map = _create_src_dest_map(srcs, dest_path)

	for src in srcs:
		scenario_destinfo_map = src_dest_map[src]
		_create_prepped_txt_file(src, scenario_destinfo_map)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
