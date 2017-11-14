import tensorflow as tf;
import numpy as np;

def instancenorm(x,bn_offset=False,bn_scale=False,name=None,eps=1e-06):
    with tf.variable_scope(name, default_name='InstanceNorm'):
        if len(x.shape) > 3:
            axises = list(np.arange(1,len(x.shape) - 1));
        else:
            axises = [1];
        mean, variance = tf.nn.moments(x,axises,name='moments');
        instshape = list(np.ones((len(x.shape),), dtype=np.int));
        instshape[0] = int(x.shape[0]);
        instshape[-1] = int(x.shape[-1]);
        mean = tf.reshape(mean,instshape);
        variance = tf.reshape(variance,instshape);
        tf.summary.histogram(name+"_mean",mean);
        tf.summary.histogram(name+"_var",variance);
        paramshape = list(np.ones((len(x.shape),), dtype=np.int));
        paramshape[-1] = int(x.shape[-1]);
        offset = None;
        if bn_offset: 
            offset_reg  = tf.contrib.layers.l2_regularizer(0.001);
            offset = tf.get_variable(shape=paramshape,initializer=tf.zeros_initializer(),regularizer=offset_reg,name=name+"_offset",dtype=tf.float32);
            tf.summary.histogram(name+"_offset",offset);
        scale = None;
        if bn_scale: 
            scale_reg  = tf.contrib.layers.l2_regularizer(0.001);
            scale = tf.get_variable(shape=paramshape,initializer=tf.zeros_initializer(),regularizer=scale_reg,name=name+"_scale",dtype=tf.float32);
            scale += tf.constant(1.0);
            tf.summary.histogram(name+"_scale",scale);
        x = tf.nn.batch_normalization(x,mean,variance,offset,scale,eps,name=name);
    return x;

def cnn_layer(x,size,stride=1,name=None,activate="relu",padding="SAME",bn_offset=False,bn_scale=False):
    W_shape = size;
    var = np.sqrt(2.0/float(size[0]*size[1]*size[2]*size[3]));
    W_init = tf.constant_initializer(np.random.normal(0.0,var,W_shape).astype(np.float32));
    W_reg  = tf.contrib.layers.l2_regularizer(0.001);
    W = tf.get_variable(shape=W_shape,initializer=W_init,regularizer=W_reg,name=name+'W');
    tf.summary.histogram(name+'W',W);
    x = tf.nn.conv2d(x,W,[1,stride,stride,1],padding,name=name+"_conv");
    x = instancenorm(x,bn_offset,bn_scale,name=name+"_instn");
    if activate is not None:
        if activate=="elu":
            x = tf.nn.elu(x,name=name+"_elu");
        elif activate=="relu":
            x = tf.nn.relu(x,name=name+"_relu");
        elif activate=="tanh":
            x = tf.tanh(x,name=name+"_tanh");
    return x;