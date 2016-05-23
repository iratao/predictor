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
import random

import tensorflow as tf

reload(sys) 
sys.setdefaultencoding('utf-8')

sess = tf.InteractiveSession()

WANT_RESTORE = True

with codecs.open('metadata.json', encoding='UTF-8') as data_file:
	jsondata = json.load(data_file)

contenttypes = jsondata["contenttypes"];
contentTypeNum = len(contenttypes);
roomtypes = jsondata["roomtypes"];
roomTypeNum = len(roomtypes);
trainingDataDir = '../designjsons';

trainingDataFiles = [item for item in os.listdir(trainingDataDir) if os.path.isfile(os.path.join(trainingDataDir, item))]
trainingDataCount = len(trainingDataFiles);
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
testDataDir = '../validators'

testDataFiles = [item for item in os.listdir(testDataDir) if os.path.isfile(os.path.join(testDataDir, item))]
testDataFiles.remove('.DS_Store')
testDataFilesCount = len(testDataFiles)

def read_test_sets_from_json(contentTypeNum, roomTypeNum, designjson):
	trainingdata = numpy.zeros((0, contentTypeNum))
	labels = numpy.zeros((0, roomTypeNum))

	countIndex = 0

	if countIndex == (len(labels)):
		logging.debug('extend labels and trainingdata')
		rownum = len(designjson)
		row = numpy.zeros((rownum, roomTypeNum))
		labels = numpy.vstack((labels, row))
		row = numpy.zeros((rownum, contentTypeNum))
		trainingdata = numpy.vstack((trainingdata, row))

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
		# init trainingdata
		for character in characters:
			for idx, contenttype in enumerate(contenttypes):
				if character['contenttype'] == contenttype['name']:
					# logging.debug('characters: idx = %d, countIndex = %d', idx, countIndex)
					trainingdata[countIndex][idx] = 1
					logging.debug('testdata[countIndex][idx] = %d', trainingdata[countIndex][idx])
					break
		countIndex += 1

	return trainingdata, labels

x = tf.placeholder(tf.float32, [None, contentTypeNum])
W = tf.Variable(tf.zeros([contentTypeNum, roomTypeNum]))
b = tf.Variable(tf.zeros([roomTypeNum]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

saver = tf.train.Saver()

y_ = tf.placeholder(tf.float32, [None, roomTypeNum])
saver.restore(sess, 'roomtype-softmax')

prediction = tf.argmax(y, 1)

from flask import Flask, jsonify, request, abort

app = Flask(__name__)


@app.route('/')
def home():
	return app.send_static_file('index.html')

@app.route('/<path:path>')
def static_proxy(path):
	# send_static_file will guess the correct MIME type
	return app.send_static_file(path)

@app.route('/home/api/v1.0/predict_example', methods=['GET'])
def nextexample():
	filename = testDataFiles[random.randint(0,  testDataFilesCount - 1)]
	with codecs.open(os.path.join(testDataDir, filename), encoding='UTF-8') as designfile:
		designjson = json.load(designfile)
	testdata, testlabel = read_test_sets_from_json(contentTypeNum, roomTypeNum, designjson)
	result = {}
	result['image'] = '/validator-img/' + filename[:-5] + '.png'
	for i in range(len(testdata)):
		index = sess.run(prediction, feed_dict={x: [testdata[i]]})[0]
		result[i] = {}
		result[i]['prediction'] = roomtypes[index]['name']
		result[i]['trueclass'] = roomtypes[numpy.argmax(testlabel[i])]['name']
		print ("Test", i, "Prediction:", roomtypes[index]['name'], ",True Class:", roomtypes[numpy.argmax(testlabel[i])]['name'])
	return jsonify(result)


@app.route('/home/api/v1.0/roomtype', methods=['POST'])
def roomtype():
    if not request.json:
        abort(400)
    contents = request.json
    testdata, testlabel = read_test_sets_from_json(contentTypeNum, roomTypeNum, contents)
    result = {}
    for i in range(len(testdata)):
		index = sess.run(prediction, feed_dict={x: [testdata[i]]})[0]
		result[i] = {}
		result[i]['prediction'] = roomtypes[index]['name']
		result[i]['trueclass'] = roomtypes[numpy.argmax(testlabel[i])]['name']
		print ("Test", i, "Prediction:", roomtypes[index]['name'], ",True Class:", roomtypes[numpy.argmax(testlabel[i])]['name'])

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)