import json

from nltk import Tree

def _gen_dependencies_tree(deps, gov_map = None, root_ix = None):

	if gov_map is None:
		gov_map = {d['governor']['index']: Tree(d['governor']['value'], []) 
				for d in deps}

	if root_ix is None:
		child_ixs = {d['dependent']['index'] for d in deps}
		governor_ixs = set(gov_map.keys()) 
		root_ix = governor_ixs.difference(child_ixs)
		root_ix = list(root_ix)[0]

	root = [d for d in deps if d['governor']['index'] is root_ix]

	for r in root:
		dep = r['dependent']
		dep_ix = dep['index']

		is_leaf = not dep_ix in gov_map

		dep_label = dep['value']
		if is_leaf:
			dep = [dep_label]
		else:
			_dependencies_to_tree(deps, gov_map, dep_ix)
			dep = gov_map[dep_ix]

		t_ix = r['governor']['index']
		t_label = r['governor']['value']

		t = gov_map[t_ix]
		rel = r['relation']
		t.append(Tree(rel['value'], [dep]))

	return t


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
	dependencies_tree = safe_load(['sentences', 'dependencies'])
	return (corefs_tree, phrase_structures_tree, dependencies_tree)
