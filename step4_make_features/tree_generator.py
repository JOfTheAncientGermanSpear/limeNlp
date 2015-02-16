import json

from nltk import Tree

def _gen_phrase_structure_tree(phrase):
	label = phrase['category']

	is_leaf = 'constituents' not in phrase
	if is_leaf:
		return label

	sorted_children = sorted(phrase['constituents'], key = lambda e: e['index'])
	sorted_children = map(_gen_phrase_structure_tree, sorted_children)
	return Tree(label, sorted_children)

def _gen_coref_tree(coref):
	def gen_mentions_tree(m):
		word_field = 'start: ' + str(m['start'])
		sent_field = 'sent: ' + str(m['sentence'])
		word_tree = Tree(word_field, [m['text']])
		return Tree(sent_field, [word_tree])

	return map(gen_mentions_tree, coref['mentions'])

def _gen_dependencies_tree():
	print('hi')

def gen_from_file(src):
	with open(src) as f:
		content = json.load(f)
		
	fn_map = {
			'coreferences': {'fn': _gen_coref_tree, 'is_array' : 1, 'id': 'id'},
			'sentences': 
				{'phrase_structure': {'fn': _gen_phrase_structure_tree},
				'dependencies': {'fn': _gen_dependencies_tree},
				'is_array': 1, 'id': 'index'}
			}

	def safe_load(field_path, fn_map = fn_map, content = content):
		field = field_path[0]
		field_path = field_path[1:]
		fn_map = fn_map[field]
		is_array = fn_map['is_array'] if 'is_array' in fn_map else 0
		if is_array:
			id_fn = lambda e: e[fn_map['id']]
			
		if field not in content:
			return None

		content = content[field]

		def gen_children(fn):
			sorted_children = sorted(content, key = id_fn)
			tree_with_id = lambda c: nltk.Tree('id: {}'.format(id_fn(c)), fn(c))
			return map(tree_with_id, sorted_children)

		if not field_path:
			fn = fn_map['fn']
			res = lambda: Tree(field, gen_children(fn)) if is_array else fn(content)
		else:
			if is_array:
				load_child = lambda c: safe_load(field_path, fn_map, c)
				res = lambda: Tree(field, gen_children(load_child))
			else:
				res = lambda: safe_load(field_path, fn_map, content)

		return res()


	corefs_tree = safe_load(['coreferences'])
	phrase_structures_tree = safe_load(['sentences', 'phrase_structure'])
	return (corefs_tree, phrase_structures_tree)