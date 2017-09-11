#!/usr/bin/env python
import os
import ssl

import gzip
import cPickle
from six.moves import urllib

def load_pickle(data_path):
	data_dir, data_file = os.path.split(data_path)
	if data_dir == "" and not os.path.isfile(data_path):
		new_path = os.path.join(os.path.split(__file__)[0], data_path)
		if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
			data_path = new_path

	# Download the MNIST dataset if it is not present
	if (not os.path.isfile(data_path)) and data_file == 'mnist.pkl.gz':
		origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
		print('Downloading data from %s' % origin)
		context = ssl._create_unverified_context()
		urllib.request.urlretrieve(origin, data_path)

	print('... loading data')

	# Load the data_path
	with gzip.open(data_path, 'rb') as f:
		try:
			data = cPickle.load(f, encoding='latin1')
		except:
			data = cPickle.load(f)

	return data
