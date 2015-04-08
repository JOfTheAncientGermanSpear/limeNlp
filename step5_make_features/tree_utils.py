from __future__ import division

import nltk
import numpy as np
from scipy import stats


def is_tree(t):
	return isinstance(t, nltk.Tree)


def shape(t):
	"""
	>>> from nltk.tree import Tree
	>>> t = Tree.fromstring("(S (NP (NNP Bob)) (VP sleeps))")
	>>> shape(t)
	(4, 2)
	"""
	return (t.height(), len(t.leaves()))


def order(t, is_root = True):
	"""
	>>> from nltk.tree import Tree
	>>> t = Tree.fromstring("(A (B b))")
	>>> order(t)
	1
	>>> order(t[0], False) #B subtree
	2
	"""
	num_parents = 0 if is_root else 1

	if not is_tree(t):
		return num_parents

	num_children = len(t)
	return num_children + num_parents
	

def orders(t, is_root = True):
	"""
	>>> from nltk.tree import Tree
	>>> t = Tree.fromstring("(A a)")
	>>> orders(t)
	array([1, 1])
	>>> t = Tree.fromstring("(A (B b))")
	>>> orders(t)
	array([1, 2, 1])
	"""
	
	o = np.array([order(t, is_root)])

	if not is_tree(t):
		return o

	def app_child_order(acc, c):
		c_orders = orders(c, is_root = False)
		return np.concatenate([acc, c_orders])

	return reduce(app_child_order, t, o)


def hierarchy(t):
	"""
	>>> from nltk.tree import Tree
	>>> o_1 = Tree.fromstring("(1)")
	>>> o_11 = Tree(11, [o_1] * 10)
	>>> r = Tree(10, [o_11] * 10)
	>>> actual = hierarchy(r)
	>>> from scipy import stats
	>>> import numpy as np
	>>> degrees = [1, 10, 11]
	>>> cnts = [100, 1, 10]
	>>> s = stats.linregress(np.log10(degrees), np.log10(cnts))[0]
	>>> s == actual
	True
	"""

	os = orders(t)
	o_counts = dict()
	for o in os:
		o_counts[o] = o_counts.get(o, 0) + 1
	
	degrees = sorted(o_counts.keys())
	cnts = [o_counts[d] for d in degrees]

	slope = stats.linregress(np.log10(degrees), np.log10(cnts))[0]
	return slope


def iota(t, is_root = True):
	"""
	>>> from nltk.tree import Tree
	>>> t = Tree.fromstring("(A (B b))")
	>>> iota(t) #1 + 2*2 + 1
	6
	>>> t = Tree.fromstring("(A (B (C (D d) (D d))))")
	>>> (a, b, c, d, d_l) = (1, 2*2, 2*3, 2*2*2, 2*1) 
	>>> iota(t) == a + b + c + d + d_l
	True
	"""

	os = orders(t, is_root)

	def iota_sum(acc, o):
		w = o if o < 2 else o * 2
		return acc + w

	return reduce(iota_sum, os, 0)


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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
