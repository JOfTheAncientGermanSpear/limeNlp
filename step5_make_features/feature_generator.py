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
	"""
	shapes_by_label = nltk.ConditionalFreqDist((s.label(), shape(s)) for s in t.subtrees())

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
	"""
	return nltk.FreqDist(s.label() for s in t.subtrees())


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
