import tensorflow as tf;
import numpy as np;
from layers import *;

def stack_blocks(x3D,x2D,HID_NUM,consts,block_num):
    for i in range(block_num//3):
        CHAN_NUM = 2*int(x2D.shape[-1]);
        if CHAN_NUM > 1024:
            CHAN_NUM = 1024;
        x3D,x2D = ResBlock(x3D,x2D,[CHAN_NUM,HID_NUM],consts,name="block_%d_0"%i);
        x3D,x2D = ResBlock(x3D,x2D,[CHAN_NUM,HID_NUM],consts,name="block_%d_1"%i);
        x3D,x2D = ResBlock(x3D,x2D,[CHAN_NUM,HID_NUM],consts,name="block_%d_2"%i);
        if int(x2D.shape[1]) > 3 and int(x2D.shape[2]) > 4:
            x2D = tf.nn.max_pool(x2D,[1,2,2,1],[1,2,2,1],"VALID");
    return x3D,x2D;

def BlockConst(size):
    BATCH_SIZE = int(size[0]);
    HID_NUM = int(size[1]);
    w0a_iv = np.zeros([BATCH_SIZE,3],np.int32);
    b0a_iv = np.zeros([BATCH_SIZE,1],np.int32);
    w0b_iv = np.zeros([BATCH_SIZE,3],np.int32);
    b0b_iv = np.zeros([BATCH_SIZE,1],np.int32);
    b1_iv = np.zeros([BATCH_SIZE,1],np.int32);
    w1_iv = np.zeros([BATCH_SIZE,3],np.int32);
    for i in range(BATCH_SIZE):
        w0a_iv[i,0] = 12*i;
        w0a_iv[i,1] = 12*i+1;
        w0a_iv[i,2] = 12*i+2;
        b0a_iv[i,0] = 12*i+3;
        w0b_iv[i,0] = 12*i+4;
        w0b_iv[i,1] = 12*i+5;
        w0b_iv[i,2] = 12*i+6;
        b0b_iv[i,0] = 12*i+7;
        b1_iv[i,0] = 12*i+8;
        w1_iv[i,0] = 12*i+9;
        w1_iv[i,1] = 12*i+10;
        w1_iv[i,2] = 12*i+11;
    w0a_idx = tf.constant(w0a_iv,shape=[BATCH_SIZE,3]);
    b0a_idx = tf.constant(b0a_iv,shape=[BATCH_SIZE,1]);
    w0b_idx = tf.constant(w0b_iv,shape=[BATCH_SIZE,3]);
    b0b_idx = tf.constant(b0b_iv,shape=[BATCH_SIZE,1]);
    b1_idx = tf.constant(b1_iv,shape=[BATCH_SIZE,1]);
    w1_idx = tf.constant(w1_iv,shape=[BATCH_SIZE,3]);
    return [w0a_idx,b0a_idx,w0b_idx,b0b_idx,b1_idx,w1_idx];

def ResBlock(x3D,x2D,size,consts,name=None):
    HEIGHT = int(x2D.shape[1]);
    WIDTH = int(x2D.shape[2]);
    CNN_HID_N = int(size[0]);
    HID_NUM = int(size[1]);
    res_w0a_idx = consts[0];
    res_b0a_idx = consts[1];
    res_w0b_idx = consts[2];
    res_b0b_idx = consts[3];
    res_b1_idx = consts[4];
    res_w1_idx = consts[5];
    with tf.name_scope(name, "ResBlock") as scope:
        x2D = cnn_layer(x2D,[3,3,int(x2D.shape[-1]),CNN_HID_N],name=name+"_x2D_cnn",bn_scale=True);
        #x2D
        tf.summary.histogram(name+"_x2D",x2D);
        if HEIGHT == 3 and WIDTH == 4:
            w = x2D;
        else:
            w = tf.nn.max_pool(x2D,[1,HEIGHT//3,WIDTH//4,1],[1,HEIGHT//3,WIDTH//4,1],"VALID");
        w = cnn_layer(w,[1,1,CNN_HID_N,HID_NUM],name=name+"_w_cnn01",bn_scale=True);
        w = cnn_layer(w,[1,1,HID_NUM,HID_NUM],name=name+"_w_cnn02",bn_offset=True,bn_scale=True,activate="linear");
        w = tf.reshape(w,[-1,HID_NUM]);
        res_w0a = tf.gather(w,res_w0a_idx);
        res_w0a = tf.reshape(res_w0a,[-1,3,HID_NUM]);
        tf.summary.histogram(name+"_res_w0a",res_w0a);
        res_b0a = tf.gather(w,res_b0a_idx);
        res_b0a = tf.reshape(res_b0a,[-1,1,HID_NUM]);
        tf.summary.histogram(name+"_res_b0a",res_b0a);
        res_w0b = tf.gather(w,res_w0b_idx);
        res_w0b = tf.reshape(res_w0b,[-1,3,HID_NUM]);
        tf.summary.histogram(name+"_res_w0b",res_w0b);
        res_b0b = tf.gather(w,res_b0b_idx);
        res_b0b = tf.reshape(res_b0b,[-1,1,HID_NUM]);
        tf.summary.histogram(name+"_res_b0b",res_b0b);
        res_b1 = tf.gather(w,res_b1_idx);
        res_b1 = tf.reshape(res_b1,[-1,1,HID_NUM]);
        tf.summary.histogram(name+"_res_b1",res_b1)
        res_w1 = tf.gather(w,res_w1_idx);
        res_w1 = tf.reshape(res_w1,[-1,HID_NUM,3]);
        tf.summary.histogram(name+"_res_w1",res_w1);
        #x3D
        res_x3Da = tf.matmul(x3D,res_w0a) + res_b0a;
        tf.summary.histogram(name+"_res_x3Da_hid",res_x3Da);
        res_x3Da = instancenorm(res_x3Da,bn_scale=True,name=name+"_instn0a");
        res_x3Db = tf.matmul(x3D,res_w0b) + res_b0b;
        tf.summary.histogram(name+"_res_x3Db_hid",res_x3Db);
        res_x3Db = instancenorm(res_x3Db,bn_scale=True,name=name+"_instn0b");
        #maxout as activation
        res_x3D = tf.maximum(res_x3Da,res_x3Db) + res_b1;
        res_x3D = tf.matmul(res_x3D,res_w1);
        res_x3D = instancenorm(res_x3D,bn_scale=True,name=name+"_instn1");
        x3D = tf.add(x3D,res_x3D,name=name+"_x3D");
        tf.summary.histogram(name+"_x3D",x3D);
    return x3D,x2D;