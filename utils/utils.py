import re

num_re = re.compile('[^0-9]*([0-9]+)[^0-9]*')
def lime_num(filename):
	""" Extract number out of a string
	>>> lime_num('LM201PD.txt')
	201
	>>> lime_num('LIME 1001 Picture Description Original.txt')
	1001
	"""
	return int(num_re.findall(filename)[0])

if __name__ == "__main__":
	import doctest
	doctest.testmod()
