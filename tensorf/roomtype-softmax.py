from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import json
import sys
import pdb
import os
import numpy
import logging

import tensorflow as tf

reload(sys) 
sys.setdefaultencoding('utf-8')

sess = tf.InteractiveSession()

IS_SIMPLE_VERSION = False
WANT_RESTORE = False

with codecs.open('metadata.json', encoding='UTF-8') as data_file:
	jsondata = json.load(data_file)

contenttypes = jsondata["contenttypes"];
contentTypeNum = len(contenttypes);
roomtypes = jsondata["roomtypes"];
roomTypeNum = len(roomtypes);
trainingDataDir = '../designjsons';

trainingDataFiles = [item for item in os.listdir(trainingDataDir) if os.path.isfile(os.path.join(trainingDataDir, item))]
trainingDataCount = len(trainingDataFiles);
logger = logging.getLogger();
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s');
testDataDir = '../validators';

testDataFiles = [item for item in os.listdir(testDataDir) if os.path.isfile(os.path.join(testDataDir, item))]


def read_data_sets(contentTypeNum, roomTypeNum, dataDir, dataFiles):
	data = numpy.zeros((0, contentTypeNum));
	labels = numpy.zeros((0, roomTypeNum))

	countIndex = 0
	for filename in testDataFiles:
		if filename == '.DS_Store':
			continue

		with codecs.open(os.path.join(testDataDir, filename), encoding='UTF-8') as designfile:
			designjson = json.load(designfile)

		
		if countIndex == (len(labels)):
			logging.debug('extend labels and data')
			rownum = len(designjson)
			row = numpy.zeros((rownum, roomTypeNum))
			labels = numpy.vstack((labels, row))
			row = numpy.zeros((rownum, contentTypeNum))
			data = numpy.vstack((data, row))

		for roomid in designjson:
			room = designjson[roomid]
			category = room['category']
			characters = room['characters']

			# init labels
			for idx, val in enumerate(roomtypes):
				if val['name'] == category:
					logging.debug('idx = %d, countIndex = %d', idx, countIndex)
					labels[countIndex][idx] = 1
					break
			# init data
			for character in characters:
				for idx, contenttype in enumerate(contenttypes):
					if character['contenttype'] == contenttype['name']:
						# logging.debug('characters: idx = %d, countIndex = %d', idx, countIndex)
						data[countIndex][idx] = 1
						logging.debug('data[countIndex][idx] = %d', data[countIndex][idx])
						break
			countIndex += 1

	return data, labels

def read_test_sets(contentTypeNum, roomTypeNum):
	logging.debug('read test sets');
	return read_data_sets(contentTypeNum, roomTypeNum, testDataDir, testDataFiles);

def read_training_sets(contentTypeNum, roomTypeNum):
	logging.debug('read training sets');
	return read_data_sets(contentTypeNum, roomTypeNum, trainingDataDir, trainingDataFiles);


# batch_xs: n by m matrix, with n(row) = the number of training exmaples and m = the number of content types
# batch_ys: n by m matrix, with n(row) = the number of training examples and m = the number of classifications (the number of room types)
batch_xs, batch_ys = read_training_sets(contentTypeNum, roomTypeNum)

x = tf.placeholder(tf.float32, [None, contentTypeNum])
W = tf.Variable(tf.zeros([contentTypeNum, roomTypeNum]))
b = tf.Variable(tf.zeros([roomTypeNum]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

saver = tf.train.Saver()

if WANT_RESTORE:
	y_ = tf.placeholder(tf.float32, [None, roomTypeNum])
	saver.restore(sess, 'roomtype-softmax')
else:
	# Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, roomTypeNum])
	cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

	tf.initialize_all_variables().run()
	# train_step.run({x: batch_xs, y_: batch_ys})
	for i in range(1000):
		# batch_xs, batch_ys = dataset.train
		train_step.run({x: batch_xs, y_: batch_ys})

	saver.save(sess, 'roomtype-softmax')


prediction = tf.argmax(y, 1)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
testdata, testlabel = read_test_sets(contentTypeNum, roomTypeNum)
print('accuracy: ')
print(sess.run(accuracy, feed_dict={x: testdata, y_: testlabel}))
for i in range(len(testdata)):
	index = sess.run(prediction, feed_dict={x: [testdata[i]]})[0]
	print ("Test", i, "Prediction:", roomtypes[index]['name'], ",True Class:", roomtypes[numpy.argmax(testlabel[i])]['name'])

# print(accuracy.eval({x: testdata, y_: testlabel}))
