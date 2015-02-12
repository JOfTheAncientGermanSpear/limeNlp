from os import listdir
from os.path import join, basename

import re

scenarios = ['circus', 'cookie_theft', 'picnic']


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

		ret_s = {}
		for k in scenarios:
			dest_filename_scenario = dest_filename.replace('l.txt', 'l_{}.txt'.format(k))
			ret_s[k] = join(dest_path, dest_filename_scenario)

		ret[s] = ret_s
	
	return ret

comment_re = re.compile(r'\W*\(.*\)\W*')

def _sanitize_content(content):
	content_lines = content.split('\n')
	sanitized_lines = [l for l in content_lines
			if len(l.split()) > 3 and
			not comment_re.match(l)]
	return ('\n').join(sanitized_lines)

def _create_prepped_txt_file(src, scenario_destpath_map):

	with open(src) as src_f:
		content = src_f.read()

	titles = dict( (s, get_title(s, content)) for s in scenarios)
		
	local_scenarios = [s for s in titles.keys() if titles[s]]

	def get_index(scenario, content = content):
		title = titles[scenario]
		return content.index(title)
	
	sorted_scenarios = sorted(local_scenarios, lambda l, r: get_index(l) - get_index(r))

	def write_scenarios(scenarios = sorted_scenarios):

		if len(scenarios) > 0:
			scenario = scenarios[0]
			tail = scenarios[1:] if len(scenarios) > 1 else []

			title = titles[scenario]

			below_scenario = content.split(title)[1]

			end_index = get_index(tail[0], below_scenario) if tail else -1
			curr_content = below_scenario[:end_index]
			curr_content = _sanitize_content(curr_content)

			write_scenarios(tail)
		else:
			return

		with open(scenario_destpath_map[scenario], 'w') as dest:
			dest.write(curr_content)

	write_scenarios()


def create_prepped_txt_files(src_path, dest_path):
	srcs = _get_original_txt_files(src_path)
	src_dest_map = _create_src_dest_map(srcs, dest_path)
	for src in src_dest_map.keys():
		scenario_destpath_map = src_dest_map[src]
		_create_prepped_txt_file(src, scenario_destpath_map)
