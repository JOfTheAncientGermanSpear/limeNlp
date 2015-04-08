from __future__ import division

import nltk
import numpy as np
from scipy import stats
import pandas as pd

import tree_utils


def is_sentence_label(l):
	"""
	>>> is_sentence_label('id: 0')
	True
	>>> is_sentence_label('id: 37')
	True
	>>> is_sentence_label('sentences')
	True
	>>> is_sentence_label('ab')
	False
	"""
	l_lower = l.lower()
	return l_lower == "sentences" or 'id: ' in l_lower


def avg_vals_fn(d):
	"""
	>>> d = {'a': [1, 2, 3], 'b': [4, 5]}
	>>> fn = avg_vals_fn(d)
	>>> fn('a')
	2.0
	>>> fn('b')
	4.5
	"""
	def fn(k):
		vals = d[k]
		return np.average([v for v in vals], axis = 0)

	return fn


def avg_metric_by_label(t, metric_fn, label_attachment):
	"""
	>>> from nltk.tree import Tree
	>>> t_0 = Tree.fromstring("(A (B b))")
	>>> t_1 = Tree.fromstring("(A a)")
	>>> fn = lambda t: t.height()
	>>> t = Tree("sentences", [Tree("id: 0", [t_0]), Tree("id: 1", [t_1])])
	>>> avg_heights = avg_metric_by_label(t, fn, '_avg_height')
	>>> sorted(avg_heights.keys())
	['A_avg_height', 'B_avg_height']
	>>> [avg_heights[l] for l in sorted(avg_heights.keys())]
	[2.5, 2.0]
	"""
	metric_by_label = nltk.ConditionalFreqDist((s.label(), metric_fn(s)) for s in t.subtrees() if not is_sentence_label(s.label()))

	avg_metric = avg_vals_fn(metric_by_label)

	def set_metric(fd, label):
		a = avg_metric(label)
		fd[label+label_attachment] = a
		return fd

	return reduce(set_metric, metric_by_label, nltk.FreqDist())


def phrase_shapes(t):
	"""
	>>> from nltk.tree import Tree
	>>> s = "(S (NP (DT The) (NN cat)) (VP (VBD ate) (NP (DT the) (NN mouse))))"
	>>> t = Tree.fromstring(s)
	>>> shapes = phrase_shapes(t)
	>>> sorted(shapes.keys())
	['DT_avg_depth', 'DT_avg_width', 'NN_avg_depth', 'NN_avg_width', 'NP_avg_depth', 'NP_avg_width', 'S_avg_depth', 'S_avg_width', 'VBD_avg_depth', 'VBD_avg_width', 'VP_avg_depth', 'VP_avg_width']
	>>> [shapes[k] for k in sorted(shapes.keys())]
	[1.0, 2.0, 1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 1.0, 2.0, 3.0, 4.0]
	>>> t = Tree("sentences", [Tree('id: 0', [t]), Tree('id: 1', [t])])
	>>> shapes = phrase_shapes(t)
	>>> sorted(shapes.keys())
	['DT_avg_depth', 'DT_avg_width', 'NN_avg_depth', 'NN_avg_width', 'NP_avg_depth', 'NP_avg_width', 'S_avg_depth', 'S_avg_width', 'VBD_avg_depth', 'VBD_avg_width', 'VP_avg_depth', 'VP_avg_width']
	>>> [shapes[k] for k in sorted(shapes.keys())]
	[1.0, 2.0, 1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 1.0, 2.0, 3.0, 4.0]
	"""
	avg_shape_by_label = avg_metric_by_label(t, tree_utils.shape, '')

	def set_w_d(fd, label):
		s = avg_shape_by_label[label]
		fd[label+'_avg_width'] = s[0]
		fd[label+'_avg_depth'] = s[1]
		return fd

	return reduce(set_w_d, avg_shape_by_label, nltk.FreqDist())


def phrase_counts(t):
	"""
	>>> from nltk.tree import Tree
	>>> s = "(S (NP (DT The) (NN cat)) (VP (VBD ate) (NP (DT the) (NN mouse))))"
	>>> t = Tree.fromstring(s)
	>>> p = phrase_counts(t)
	>>> sorted(p.keys())
	['DT', 'NN', 'NP', 'S', 'VBD', 'VP']
	>>> [p[k] for k in sorted(p.keys())]
	[2, 2, 2, 1, 1, 1]
	>>> t = Tree("sentences", [Tree('id: 0', [t]), Tree('id: 1', [t])])
	>>> p = phrase_counts(t)
	>>> sorted(p.keys())
	['DT', 'NN', 'NP', 'S', 'VBD', 'VP']
	>>> [p[k] for k in sorted(p.keys())]
	[4, 4, 4, 2, 2, 2]
	"""
	return nltk.FreqDist(s.label() for s in t.subtrees() if not is_sentence_label(s.label()))


def phrase_sentence_cover(t, coeff = 1.0, covers = dict()):
	"""
	>>> from nltk.tree import Tree
	>>> s = "(A (B (C c) (D d)) (E e))"
	>>> t = Tree.fromstring(s)
	>>> c = phrase_sentence_cover(t)
	>>> sorted(c.keys())
	['A', 'B', 'C', 'D', 'E']
	>>> [c[k] for k in sorted(c.keys())]
	[1.0, 0.5, 0.25, 0.25, 0.5]
	>>> s_small = "(A (B (C c) (D d)))"
	>>> t_small = Tree.fromstring(s_small)
	>>> t = Tree("sentences", [Tree("id: 0", [t]), Tree("id: 1", [t_small])])
	>>> c = phrase_sentence_cover(t)
	>>> sorted(c.keys())
	['A', 'B', 'C', 'D', 'E']
	>>> [c[k] for k in sorted(c.keys())]
	[1.0, 0.75, 0.375, 0.375, 0.5]
	"""
	if not tree_utils.is_tree(t):
		return covers

	label = t.label()

	if is_sentence_label(label):
		covers_per_sent = map(lambda s: phrase_sentence_cover(s, coeff, dict()), t)
		return tree_utils.avg_dicts(covers_per_sent)

	covers[label] = covers.get(label, 0) + coeff

	num_children = len(t)

	for c in t:
		phrase_sentence_cover(c, coeff/num_children, covers)

	return covers


def phrase_ratios(p):
	"""
	>>> from nltk.tree import Tree
	>>> s = "(S (NP (DT The) (NN cat)) (VP (VBD ate) (NP (DT the) (NN mouse))))"
	>>> t = Tree.fromstring(s)
	>>> p = phrase_ratios(t)
	>>> sorted(p.keys())
	['DT', 'NN', 'NP', 'S', 'VBD', 'VP']
	>>> vals = [p[k] for k in sorted(p.keys())]
	>>> vals == [2.0/9, 2.0/9, 2.0/9, 1.0/9, 1.0/9, 1.0/9] 
	True
	"""
	if tree_utils.is_tree(p):
		p = phrase_counts(p)
	
	ratios = nltk.FreqDist()
	for l in p:
		ratios[l] = p.freq(l)
	
	return ratios


def phrase_hierarchies(t):
	"""
	>>> from nltk.tree import Tree
	>>> s = "(S (NP (DT The) (NN cat)) (VP (VBD ate) (NP (DT the) (NN mouse))))"
	>>> t = Tree.fromstring(s)
	>>> hiers = phrase_hierarchies(t)
	>>> sorted(hiers.keys())
	['DT_avg_hier', 'NN_avg_hier', 'NP_avg_hier', 'S_avg_hier', 'VBD_avg_hier', 'VP_avg_hier']
	>>> t = Tree("sentences", [Tree('id: 0', [t]), Tree('id: 1', [t])])
	>>> hiers = phrase_hierarchies(t)
	>>> sorted(hiers.keys())
	['DT_avg_hier', 'NN_avg_hier', 'NP_avg_hier', 'S_avg_hier', 'VBD_avg_hier', 'VP_avg_hier']
	"""

	return avg_metric_by_label(t, tree_utils.hierarchy, '_avg_hier')


def phrase_iotas(t):
	"""
	>>> from nltk.tree import Tree
	>>> s = "(S (NP (DT The) (NN cat)) (VP (VBD ate) (NP (DT the) (NN mouse))))"
	>>> t = Tree.fromstring(s)
	>>> iotas = phrase_iotas(t)
	>>> sorted(iotas.keys())
	['DT_avg_iota', 'NN_avg_iota', 'NP_avg_iota', 'S_avg_iota', 'VBD_avg_iota', 'VP_avg_iota']
	>>> t = Tree("sentences", [Tree('id: 0', [t]), Tree('id: 1', [t])])
	>>> iotas = phrase_iotas(t)
	>>> sorted(iotas.keys())
	['DT_avg_iota', 'NN_avg_iota', 'NP_avg_iota', 'S_avg_iota', 'VBD_avg_iota', 'VP_avg_iota']
	"""

	return avg_metric_by_label(t, tree_utils.iota, '_avg_iota')


def phrase_feature_matrix(t):
	def labeled_series(fd, l):
		return pd.Series({k + '_' + l: fd[k] for k in fd})

	counts = labeled_series(phrase_counts(t), 'count')
	ratios = labeled_series(phrase_ratios(t), 'ratio')
	shapes = pd.Series(phrase_shapes(t))
	hierarchy = pd.Series(phrase_hierarchies(t))
	iota = pd.Series(phrase_iotas(t))

	return pd.concat([counts, ratios, shapes, hierarchy, iota])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
