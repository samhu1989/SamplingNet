import tensorflow as tf;
import numpy as np;
from layers import *;

def GMM(unit_gmm):
    print unit_gmm.shape;
    basis_num = int(unit_gmm.shape[0]);
    pts_num = 1024;
    pts_dim = int(unit_gmm.shape[2]);
    with tf.variable_scope("GMM") as scope:
        try:
            mu_shape = [basis_num,pts_num,pts_dim];
            mu_init  = tf.constant_initializer(np.random.normal(0.0,0.5,mu_shape).astype(np.float32));
            mu = tf.get_variable(shape=mu_shape,initializer=mu_init,trainable=True,name='mu');
            var_shape = [basis_num,pts_num,pts_dim,pts_dim];
            var_init  = tf.constant_initializer(np.random.normal(0.0,0.5,var_shape).astype(np.float32));
            var = tf.get_variable(shape=var_shape,initializer=var_init,trainable=True,name='var');
        except ValueError:
            scope.reuse_variables();
            mu = tf.get_variable("mu");
            var = tf.get_variable("var");
    unit_gmm = tf.reshape(unit_gmm,[basis_num,-1,pts_dim,1]);
    var = var + tf.transpose( var , perm=[0, 1, 3 ,2]);
    pts = tf.matmul( var, unit_gmm ); 
    pts = tf.reshape(pts,[basis_num,-1,pts_dim]);
    pts += mu;
    return pts;
    
def MorphableModel(w,unit_gmm):
    w = tf.reshape(w,[int(w.shape[0]),-1]);
    tf.summary.histogram("MorphableModel_w",w);
    basis = GMM(unit_gmm);
    shape = basis.shape.as_list();
    shape[0] = int(w.shape[0]);
    basis = tf.reshape(basis,[int(basis.shape[0]),-1]);
    tf.summary.histogram("MorphableModel_basis",basis);
    pts = tf.matmul(w,basis);
    pts = tf.reshape(pts,shape);
    tf.summary.histogram("MorphableModel_pts",pts);
    return pts;
    
    