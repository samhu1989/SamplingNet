import tensorflow as tf;
import numpy as np;
import tflearn;
import math;
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

def kcnn_layer(x3D,knn_index,name="kcnn"):
    batch_size = int(x3D.shape[0]);
    pts_num = int(x3D.shape[1]);
    k = int(knn_index.shape[-2]);
    x3Dkcnn = tf.gather_nd( x3D , knn_index , name = name+"_gather" );
    #centralization
    x3Dkcnn -= tf.reshape(x3D,[batch_size,pts_num,1,3]);
    num = int(2**math.ceil(math.log(k*3,2)));
    print num;
    x3Dkcnn = tflearn.layers.conv.conv_2d(x3Dkcnn,num,(1,1),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name=name+"_cnn_a");
    x3Dkcnn = tflearn.layers.conv.conv_2d(x3Dkcnn,num//2,(1,1),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"_cnn_b");
    x3Dkcnn = tf.reduce_max(x3Dkcnn,2);
    x3Dkcnn = tf.reshape(x3Dkcnn,[batch_size,pts_num,1,-1]);
    x3Dkcnn = tflearn.layers.conv.conv_2d(x3Dkcnn,num//4,(1,1),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name=name+"_cnn_c");
    x3Dkcnn = tflearn.layers.conv.conv_2d(x3Dkcnn,3,(1,1),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"_cnn_d");
    x3Dkcnn = tf.reshape(x3Dkcnn,[batch_size,pts_num,3]);
    return x3D + x3Dkcnn;

def fc_to_param(x2D,x1D,batch_size,k,name="fc_to_param"):
    param = [];
    num = int(2**math.ceil(math.log(k*3,2)));
    n = [(num//2)*(num//4),num//4,(num//4)*3,3]; 
    shape = [ [batch_size,num//2,num//4] , [batch_size,1,num//4]  , [batch_size,num//4,3] , [batch_size,1,3] ];
    x1D = tflearn.layers.core.fully_connected(x1D,1024,activation='linear',weight_decay=1e-4,regularizer='L2')
    x1D = tf.nn.relu( tf.add( x1D, tflearn.layers.core.fully_connected(x2D,1024,activation='linear',weight_decay=1e-3,regularizer='L2') ) );
    for i in range(len(n)):
        x1D = tflearn.layers.core.fully_connected( x1D,n[i],activation='relu',weight_decay=1e-4,regularizer='L2');
        x1D = tflearn.layers.core.fully_connected( x1D,n[i],activation='linear',weight_decay=1e-4,regularizer='L2');
        param.append(tf.reshape(x1D,shape[i]));
    return param;
    
def param_kcnn_layer(x3D,knn_index,param_lst,name="param_kcnn"):
    batch_size = int(x3D.shape[0]);
    k = int(knn_index.shape[-2]);
    x3Dkcnn = tf.gather_nd( x3D , knn_index , name = name+"_gather" );
    num = int(2**math.ceil(math.log(k*3,2)));
    x3Dkcnn = tflearn.layers.conv.conv_2d(x3Dkcnn,num,(1,1),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',name=name+"_cnn_a");
    x3Dkcnn = tflearn.layers.conv.conv_2d(x3Dkcnn,num//2,(1,1),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2',name=name+"_cnn_b");
    x3Dkcnn = tf.reduce_max(x3Dkcnn,2);
    x3Dkcnn = tf.add(tf.matmul(x3Dkcnn,param_lst[0],name=name+"_matmul0"),param_lst[1],name=name+"_add0");
    x3Dkcnn = tf.nn.relu( x3Dkcnn );
    x3Dkcnn = tf.add(tf.matmul(x3Dkcnn,param_lst[2],name=name+"_matmul1"),param_lst[3],name=name+"_add1");
    return x3D + x3Dkcnn;
    
    