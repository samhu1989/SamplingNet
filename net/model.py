from block import *;
import loss;
import tensorflow as tf;
from psgn import *;
from vpsgn import *;
from vkpsgn import *;
from kparam import *;
from phase import *;

def build_model(name,phase=None,size=None):
    return eval(name)(phase,size);

def TemplateNet(phase,size=[16,3,128,192,256]):
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
        yGT = tf.placeholder(tf.float32,shape=[None,PTS_DIM],name='yGT');
        x3D = tf.placeholder(tf.float32,shape=[None,PTS_DIM],name='x3D');
        ins.append(yGT);
        ins.append(x3D);
        yGT = tf.reshape(yGT,[BATCH_SIZE,-1,PTS_DIM]);
        x3D = tf.reshape(x3D,[HID_NUM,-1,PTS_DIM]);
        x2D = tf.placeholder(tf.float32,shape=[None,HEIGHT,WIDTH,4],name='x2D');
        x2D = tf.reshape(x2D,[BATCH_SIZE,HEIGHT,WIDTH,4]);
        ins.append(x2D);
        features = fcn.FCN(x2D,HID_NUM);
        affine = fcn.predictAffine( features[-2] );
        x3Dref = tf.truncated_normal( [HID_NUM,1024,PTS_DIM] , mean=0.0 , stddev=1.0 , dtype=tf.float32 ,name="x3Dref" );
        
        x3D = gmm.MorphableModel( features[-1],x3D );
        #x3D = tf.matmul(x3D,affine);
        
        x3Dref = gmm.MorphableModel( features[-1] , x3Dref );
        #x3Dref = tf.matmul(x3Dref,affine);
        
        outs.append(x3D);
        forwardD,_,backwardD,_ = loss.ChamferDistLoss.Loss(x3D,yGT);
        forwardDref,_,backwardDref,_ = loss.ChamferDistLoss.Loss(x3Dref,yGT);
        D = tf.reduce_sum( forwardD + backwardD , 1 );
        tf.summary.histogram("D",D);
        Dref = tf.reduce_sum( forwardDref + backwardDref , 1);
        tf.summary.histogram("Dref",D);
        minDof2 = tf.minimum( D , Dref );
        minDof2 = tf.reduce_mean( minDof2 );
        loss_nodecay = minDof2 * 100 ;
        tf.summary.scalar("loss_nodecay",loss_nodecay);
        outs.append(loss_nodecay);
        decay = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.001;
        tf.summary.scalar("decay",decay);
        loss_with_decay = tf.add(loss_nodecay,decay,name="loss");
        tf.summary.scalar("loss_with_decay",loss_with_decay);
        outs.append(loss_with_decay);
        lr  = tf.placeholder(tf.float32,name='lr');
        ins.append(lr);
        gstep = tf.get_variable(shape=[],initializer=tf.constant_initializer(0),trainable=False,name='step',dtype=tf.int32);
        outs.append(gstep);
        opt0 = tf.train.AdamOptimizer(lr).minimize(loss_with_decay,global_step=gstep);
        opt_ops.append(opt0);
        sum_ops.append(tf.summary.merge_all());
    return ins,outs,opt_ops,sum_ops;

def ParamNet(phase,size=[16,3,256,192,256]):
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
        yGT = tf.placeholder(tf.float32,shape=[None,PTS_DIM],name='yGT');
        x3D = tf.placeholder(tf.float32,shape=[None,PTS_DIM],name='x3D');
        ins.append(yGT);
        ins.append(x3D);
        yGT = tf.reshape(yGT,[BATCH_SIZE,-1,PTS_DIM]);
        x3D = tf.reshape(x3D,[BATCH_SIZE,-1,PTS_DIM]);
        x2D = tf.placeholder(tf.float32,shape=[None,HEIGHT,WIDTH,4],name='x2D');
        x2D = tf.reshape(x2D,[BATCH_SIZE,HEIGHT,WIDTH,4]);
        ins.append(x2D);
        consts = block.BlockConst([BATCH_SIZE,HID_NUM]);
        x3D,x2D = block.stack_blocks(x3D,x2D,HID_NUM,consts,27);
        outs.append(x3D);
        forwardD,_,backwardD,_ = loss.ChamferDistLoss.Loss(x3D,yGT);
        loss_correspondent = 100 * tf.reduce_mean( tf.reduce_sum( tf.reduce_sum( tf.square( x3D - yGT ) , 2 ) , 1 ) );
        tf.summary.scalar("loss_correspondent",loss_correspondent);
        forwardD = tf.reduce_mean(forwardD);
        backwardD = tf.reduce_mean(backwardD);
        loss_nodecay = ( forwardD + backwardD ) * 2048 * 100 ;
        tf.summary.scalar("loss_nodecay",loss_nodecay);
        outs.append(loss_nodecay);
        decay = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.01;
        tf.summary.scalar("decay",decay);
        loss_with_decay = tf.add(loss_nodecay,decay,name="loss");
        tf.summary.scalar("loss",loss_with_decay);
        outs.append(loss_with_decay);
        lr  = tf.placeholder(tf.float32,name='lr');
        ins.append(lr);
        gstep = tf.get_variable(shape=[],initializer=tf.constant_initializer(0),trainable=False,name='step',dtype=tf.int32);
        outs.append(gstep);
        opt0 = tf.train.AdamOptimizer(lr).minimize(loss_with_decay,global_step=gstep);
        opt_ops.append(opt0);
        opt1 = tf.train.AdamOptimizer(lr).minimize(loss_correspondent,global_step=gstep);
        opt_ops.append(opt1);
        sum_ops.append(tf.summary.merge_all());
    return ins,outs,opt_ops,sum_ops;