import tensorflow as tf;
import numpy as np;
import loss;
import tflearn;
from phase import*;

def RPSGN(phase,size=[16,3,128,192,256]):
    BATCH_SIZE=size[0];
    PTS_DIM=size[1];
    HID_NUM=size[2];
    HEIGHT=size[3];
    WIDTH=size[4];
    if phase == PHASE.TRAIN:
        dev='/gpu:0';
    elif phase == PHASE.TEST:
        dev='/cpu:0';
    else:
        raise Exception("Invalid phase");
    ins=[];
    outs=[];
    opt_ops=[];
    sum_ops=[];
    with tf.device( dev ):
        tflearn.init_graph(seed=1029,num_cores=1,gpu_memory_fraction=0.9,soft_placement=True)
        yGT = tf.placeholder(tf.float32,shape=[None,PTS_DIM],name='yGT');
        x3D = tf.placeholder(tf.float32,shape=[None,PTS_DIM],name='x3D');
        ins.append(yGT);
        ins.append(x3D);   
        yGT = tf.reshape(yGT,[BATCH_SIZE,-1,PTS_DIM]);
        x3D = tf.reshape(x3D,[HID_NUM,-1,PTS_DIM]);
        x2D = tf.placeholder(tf.float32,shape=[None,HEIGHT,WIDTH,4],name='x2D');
        x2D = tf.reshape(x2D,[BATCH_SIZE,HEIGHT,WIDTH,4]);
        ins.append(x2D);
        x = x2D;
        
        r2D = tf.placeholder(tf.float32,shape=[BATCH_SIZE,512],name='r2D');
        ins.append(r2D);
        
        r2D = tflearn.layers.core.fully_connected(r2D,1024,activation='relu',weight_decay=1e-3,regularizer='L2');
        r2D = tflearn.layers.core.fully_connected(r2D,3072,activation='relu',weight_decay=1e-3,regularizer='L2');
        r2D = tf.reshape(r2D,[BATCH_SIZE,48,64,1]);
        r2D = tflearn.layers.conv.conv_2d_transpose(r2D,1,[5,5],[96,128],strides=2,activation='relu',weight_decay=1e-5,regularizer='L2');
        r2D = tflearn.layers.conv.conv_2d(r2D,1,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2');
#192 256
        x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2');
        x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2');
        x0=x
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#96 128
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x1=x
        x = tf.concat([x,r2D],3);
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#48 64
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x2=x
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#24 32
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x3=x
        x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#12 16
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x4=x
        x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#6 8
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x5=x
        x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#3 4
        x_additional=tflearn.layers.core.fully_connected(x,2048,activation='relu',weight_decay=1e-3,regularizer='L2')
        x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[6,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#6 8
        x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x5))
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x5=x  
        x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[12,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#12 16
        x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x4))
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x4=x
        x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[24,32],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#24 32
        x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x3))
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x3=x
        x=tflearn.layers.conv.conv_2d_transpose(x,32,[5,5],[48,64],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#48 64
        x2=tflearn.layers.conv.conv_2d(x2,32,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x2))
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x2=x
        x=tflearn.layers.conv.conv_2d_transpose(x,16,[5,5],[96,128],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#96 128
        x1=tflearn.layers.conv.conv_2d(x1,16,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x1))
        x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#48 64
        x2=tflearn.layers.conv.conv_2d(x2,32,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x2))
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x2=x
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#24 32
        x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x3))
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x3=x
        x=tflearn.layers.conv.conv_2d(x,128,(5,5),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#12 16
        x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x4))
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x4=x
        x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#6 8
        x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x5))
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x5=x
        x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#3 4
        x_additional=tflearn.layers.core.fully_connected(x_additional,2048,activation='linear',weight_decay=1e-4,regularizer='L2')
        x_additional=tf.nn.relu(tf.add(x_additional,tflearn.layers.core.fully_connected(x,2048,activation='linear',weight_decay=1e-3,regularizer='L2')))
        x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[6,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#6 8
        x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x5))
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x5=x  
        x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[12,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#12 16
        x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x4))
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x4=x
        x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[24,32],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#24 32
        x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x3))
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')

        x_additional=tflearn.layers.core.fully_connected(x_additional,1024,activation='relu',weight_decay=1e-3,regularizer='L2')
        x_additional=tflearn.layers.core.fully_connected(x_additional,256*3,activation='linear',weight_decay=1e-3,regularizer='L2')
        x_additional=tf.reshape(x_additional,(BATCH_SIZE,256,3))
        x=tflearn.layers.conv.conv_2d(x,3,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.reshape(x,(BATCH_SIZE,32*24,3))
        x=tf.concat([x_additional,x],1)
        x=tf.reshape(x,(BATCH_SIZE,-1,3))
        outs.append(x);
        dists_forward,_,dists_backward,_ = loss.ChamferDistLoss.Loss(yGT,x)
        dists_forward = tf.reduce_mean(dists_forward,1);
        dists_backward = tf.reduce_mean(dists_backward,1);
        dists = dists_forward + dists_backward;
        outs.append(dists);
        dists_forward = tf.reduce_mean(dists_forward);
        dists_backward = tf.reduce_mean(dists_backward);
        tf.summary.scalar("dists_forward",dists_forward);
        tf.summary.scalar("dists_backward",dists_backward);
        loss_nodecay = ( dists_forward + dists_backward )*1024*100;
        tf.summary.scalar("loss_no_decay",loss_nodecay);
        decay = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.1;
        tf.summary.scalar("decay",decay);
        outs.append(loss_nodecay);
        loss_with_decay = loss_nodecay + decay;
        tf.summary.scalar("loss_with_decay",loss_with_decay);
        outs.append(loss_with_decay);
        lr  = tf.placeholder(tf.float32,name='lr');
        ins.append(lr);
        gstep = tf.get_variable(shape=[],initializer=tf.constant_initializer(0),trainable=False,name='step',dtype=tf.int32);
        outs.append(gstep);
        opt0 = tf.train.AdamOptimizer(lr).minimize(loss_with_decay,global_step=gstep)
        opt_ops.append(opt0);
        sum_ops.append(tf.summary.merge_all());
        return ins,outs,opt_ops,sum_ops;