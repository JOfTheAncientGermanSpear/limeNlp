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
	return re.compile(r'\n\W*({})\W*\n'.format(scenario), re.IGNORECASE)

title_res = {}
title_res['circus'] = mk_title_re('circus')
title_res['cookie_theft'] = mk_title_re('cooki?e theft')
title_res['picnic'] = mk_title_re('picnic')

def get_title(scenario, content):
	ret = title_res[scenario].findall(content)
	return ret[0] if ret else None


def _get_original_txt_files(path):
	f_names = [f for f in listdir(path) if f.endswith('Original.txt')]
	return [join(path, f) for f in f_names]


def _create_src_dest_map(src_paths, dest_path):
	ret = {}
	for s in src_paths:
		src_filename = basename(s)
		dest_filename = src_filename.replace(' ', '_')
		dest_filename = dest_filename.lower()
		dest_filename = join(dest_path, dest_filename)

		ret_s = {}
		for k in scenarios:
			dest_content_filename = dest_filename.replace('l.txt', 'l_{}.txt'.format(k))
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

	titles = dict( (s, get_title(s, content)) for s in scenarios)
		
	local_scenarios = [s for s in titles.keys() if titles[s]]

	def get_word_index(scenario, content = content):
		title = titles[scenario]
		return content.index(title)
	
	sorted_scenarios = sorted(local_scenarios, lambda l, r: get_word_index(l) - get_word_index(r))

	def write_scenarios(scenarios = sorted_scenarios):

		if len(scenarios) > 0:
			scenario = scenarios[0]
			tail = scenarios[1:] if len(scenarios) > 1 else []

			title = titles[scenario]

			below_scenario = content.split(title)[1]

			end_index = get_word_index(tail[0], below_scenario) if tail else -1
			curr_content = below_scenario[:end_index]
			curr_content, sanitize_meta = _sanitize_content(curr_content)

			write_scenarios(tail)
		else:
			return

		destinfo = scenario_destinfo_map[scenario]
		destinfo.write(content = curr_content, meta = sanitize_meta)


	write_scenarios()


def create_prepped_txt_files(src_path, dest_path):
	srcs = _get_original_txt_files(src_path)
	src_dest_map = _create_src_dest_map(srcs, dest_path)
	for src in src_dest_map.keys():
		scenario_destinfo_map = src_dest_map[src]
		_create_prepped_txt_file(src, scenario_destinfo_map)
