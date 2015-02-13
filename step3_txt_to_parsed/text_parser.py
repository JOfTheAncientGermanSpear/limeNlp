from __future__ import division

import os
from os import listdir
from os.path import basename, join, splitext

import requests

base_url = 'http://localhost:9000/api/parse/'

def parse(text):
	url = base_url + text.strip()
	response = requests.get(url)
	return response.json()

def parse_to_file(src_abs_filename, dest_abs_filename):
	with open(src_abs_filename) as src:
		content = src.read()

	if not content:
		return

	parsed = parse(content)

	with open(dest_abs_filename, 'w') as dest:
		dest.write(str(parsed))
		
def parse_to_dir(src_dir, dest_dir, src_filter = lambda f: f.endswith('.txt')):
	if not os.path.exists(dest_dir):
		os.makedirs(dest_dir)

	src_files = [join(src_dir, f) for f in listdir(src_dir) if src_filter(f)]
	num_files = len(src_files)
	for (i, s) in enumerate(src_files):
		print('file: {}, percent_done: {}'.format(basename(s), i/num_files * 100))
		d = splitext(basename(s))[0] + '.json'
		d = join(dest_dir, d)
		parse_to_file(s, d)
