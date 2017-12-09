# -*- coding: utf-8 -*-
import tensorflow as tf;
import os;
from tensorflow.python.framework import ops
path = os.path.dirname(os.path.realpath(__file__));
group_module=tf.load_op_library(path+'/group.so');

def group(xyz,k):
	'''
Computes the distance of k nearest neighbors inside a point clouds
output: dist:  (batch_size,#point,k)   nearest k neighbor dist inside point set 
output: idx:  (batch_size,#point,k)   nearest k neighbor index inside point set
	'''
	return group_module.knn(xyz,k);


if __name__=='__main__':
	import numpy as np;
	import random;
	import time;
	xyz=np.random.randn(2,256,3).astype('float32');
	with tf.Session('') as sess:
		with tf.device('/cpu:0'):
			ixyzcpu=tf.Variable(xyz)
			_,idxcpu=group(ixyzcpu,8);
		sess.run(tf.global_variables_initializer())
		t0=time.time();
		for i in xrange(100):
			valcpu = sess.run(idxcpu);
		valcpu = sess.run(idxcpu);
		cputime = time.time()-t0;
		with tf.device('/gpu:0'):
			ixyzgpu=tf.Variable(xyz)
			_,idxgpu=group(ixyzgpu,8);
		sess.run(tf.global_variables_initializer())
		t0=time.time();
		for i in xrange(100):
			valgpu = sess.run(idxgpu);
		valgpu = sess.run(idxgpu);
		gputime = time.time()-t0;
	print "cputime",cputime;
	print "gputime",gputime;
	print "xyz:",xyz.shape;
	print xyz
	print "valgpu:",valgpu.shape;
	print valgpu;
	print "valcpu:",valcpu.shape;
	print valcpu;
	print "(valgpu==valcpu) is ",(valgpu==valcpu).all();
	
