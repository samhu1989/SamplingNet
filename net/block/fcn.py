import tensorflow as tf;
import numpy as np;
from layers import *;

def FCN(x2D,out_dim=32,max_channel=1024):
    cnt = 0;
    features = [];
    x2D = cnn_layer(x2D,[3,3,int(x2D.shape[-1]),8],name="x2D_cnn_%d"%cnt,bn_scale=True);
    while x2D.shape[-3] > 1 and x2D.shape[-2] > 1:
        if int(x2D.shape[-1]) < max_channel:
            ch = 2*int(x2D.shape[-1]);
        else:
            ch = int(x2D.shape[-1]);
        x2D = cnn_layer(x2D,[3,3,int(x2D.shape[-1]),ch],name="x2D_cnn_%d_a"%cnt,bn_scale=True);
        x2D = cnn_layer(x2D,[3,3,int(x2D.shape[-1]),int(x2D.shape[-1])],name="x2D_cnn_%d_b"%cnt,bn_scale=True);
        x2D = cnn_layer(x2D,[3,3,int(x2D.shape[-1]),int(x2D.shape[-1])],name="x2D_cnn_%d_c"%cnt,bn_scale=True,activate="linear");
        x2D = tf.nn.softmax(x2D,name="x2D_softmax_%d"%cnt);
        x2D = tf.nn.max_pool(x2D,[1,2,2,1],[1,2,2,1],"VALID",name="max_pool_%d"%cnt);
        features.append(x2D);
        tf.summary.histogram("x2D_cnn_%d"%cnt,x2D);
        cnt += 1;
    x2D = cnn_layer(x2D,[1,1,int(x2D.shape[-1]),( out_dim + int(x2D.shape[-1]) )/2 ],name="x2D_cnn_%d_a"%cnt,bn_scale=True,activate="linear");
    x2D = tf.nn.softmax(x2D,name="x2D_softmax_%d_a"%cnt);
    tf.summary.histogram("x2D_cnn_%d_a"%cnt,x2D);
    x2D = tf.reshape(x2D,[int(x2D.shape[0]),-1]);
    W_shape = [int(x2D.shape[-1]),out_dim];
    W_init  = tf.constant_initializer(np.random.normal(0.0,0.5,W_shape).astype(np.float32));
    W = tf.get_variable(shape=W_shape,initializer=W_init,trainable=True,name='FCN_W');
    x2D = tf.matmul(x2D,W);
    x2D = tf.nn.softmax(x2D,name="x2D_softmax_%d_b"%cnt);
    features.append(x2D);
    return features;

def predictAffine(x2D):
    x2D = cnn_layer(x2D,[1,1,int(x2D.shape[-1]),int(x2D.shape[-1])/2],name="Affine_x2D_cnn_a",bn_scale=True);
    x2D = cnn_layer(x2D,[1,1,int(x2D.shape[-1]),9],name="Affine_x2D_cnn_b",bn_scale=True,bn_offset=True,activate="linear");
    x2D = tf.nn.max_pool(x2D,[1,int(x2D.shape[-3]),int(x2D.shape[-2]),1],[1,int(x2D.shape[-3]),int(x2D.shape[-2]),1],"VALID",name="Affine_max_pool");
    affine = tf.reshape(x2D,[int(x2D.shape[0]),3,3]);
    return affine;
    
    