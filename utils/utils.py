import re

num_re = re.compile('.*?([0-9]{3,}).*')
def lime_num(filename):
	""" Extract number out of a string
	>>> lime_num('LM201PD.txt')
	201
	>>> lime_num('LIME 1002 Picture Description Original.txt')
	1002
	>>> lime_num('abcdef.txt')

	>>> lime_num('/users/data/1_abu/bc/LM401.txt')
	401
	>>> lime_num('LIME 1002 4Picture Description Original.txt')
	1002
	"""
	matches = num_re.findall(filename)
	return int(matches[0]) if matches else None

def within_bounds(f, bounds):
	"""
	>>> within_bounds('lime_1030_description', [1029, 1031])
	True
	>>> within_bounds('lime_1030_description', [1029, 1030])
	True
	>>> within_bounds('lime_1030_description', [1027, 1029])
	False
	>>> within_bounds('lime_1030_description', [1037, 1029])
	True
	"""
	f_num = lime_num(f)
	bounds = sorted(bounds)
	return f_num >= bounds[0] and f_num <= bounds[1]


if __name__ == "__main__":
	import doctest
	doctest.testmod()
