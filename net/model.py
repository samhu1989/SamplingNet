from block import *;
import loss;
import tensorflow as tf;
class Enum(set):
    def __getattr__(self, name):
        if name in self:
            return name;
        raise AttributeError;
PHASE = Enum(["TRAIN","TEST"]);

def build_model(phase):
    BATCH_SIZE=16;
    PTS_DIM=3;
    HID_NUM=256;
    HEIGHT=192;
    WIDTH=256;
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
        forwardD,_,backwardD,_ = loss.ChamferDistLoss.Loss(x3D,yGT);
        forwardD = tf.reduce_mean(forwardD);
        backwardD = tf.reduce_mean(backwardD);
        lossnodecay = ( ( forwardD + backwardD ) / 2.0 ) * 10000.0;
        lr  = tf.placeholder(tf.float32,name='lr');
        print x2D.shape;
        outs.append(x3D);
        gstep = tf.get_variable(shape=[],initializer=tf.constant_initializer(0),trainable=False,name='step',dtype=tf.int32);
        gstep_op = gstep.assign_add(1);
        sum_ops.append(tf.summary.merge_all());
    return ins,outs,opt_ops,sum_ops;