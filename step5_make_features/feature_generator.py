from __future__ import division

import nltk
import numpy as np
import pandas as pd

def shape(t):
	"""
	>>> from nltk.tree import Tree
	>>> t = Tree.fromstring("(S (NP (NNP Bob)) (VP sleeps))")
	>>> shape(t)
	(4, 2)
	"""
	return (t.height(), len(t.leaves()))

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
	shapes_by_label = nltk.ConditionalFreqDist((s.label(), shape(s)) for s in t.subtrees() if not is_sentence_label(s.label()))

	def avg_shape(label):
		shapes = shapes_by_label[label]
		return tuple(np.average([s for s in shapes], axis = 0))

	def update_w_d(fd, label):
		s = avg_shape(label)
		fd[label+'_avg_width'] = s[0]
		fd[label+'_avg_depth'] = s[1]
		return fd

	return reduce(update_w_d, shapes_by_label, nltk.FreqDist())

	return nltk.FreqDist({l:avg_shape(l) for l in shapes_by_label})


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

def avg_dicts(ds):
	"""
	>>> d_a = {'a': 3, 'b': 4}
	>>> d_b = {'a': 4, 'b': 5, 'c': 3}
	>>> d_avg = avg_dicts([d_a, d_b]) 
	>>> [d_avg[k] for k in sorted(d_avg.keys())]
	[3.5, 4.5, 3.0]
	"""
	key_counts = dict()

	def running_avg(avg, d):
		for k in d:
			prev_count = key_counts.get(k, 0)
			curr_count = prev_count + 1
			key_counts[k] = curr_count

			prev_avg = avg.get(k, 0)
			avg[k] = (prev_avg * prev_count + d[k])/curr_count
		return avg

	return reduce(running_avg, ds, dict())

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
	if not isinstance(t, nltk.Tree):
		return covers

	label = t.label()

	if is_sentence_label(label):
		covers_per_sent = map(lambda s: phrase_sentence_cover(s, coeff, dict()), t)
		return avg_dicts(covers_per_sent)

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
	if isinstance(p, nltk.Tree):
		p = phrase_counts(p)
	
	ratios = nltk.FreqDist()
	for l in p:
		ratios[l] = p.freq(l)
	
	return ratios


def phrase_feature_matrix(t):
	def labeled_series(fd, l):
		return pd.Series({k + '_' + l: fd[k] for k in fd})

	counts = labeled_series(phrase_counts(t), 'count')
	ratios = labeled_series(phrase_ratios(t), 'ratio')
	shapes = pd.Series(phrase_shapes(t))

	return pd.concat([counts, ratios, shapes])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
