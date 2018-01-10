import net;
import sys;
import os;
import shutil;
import tensorflow as tf;
import util;
import numpy as np;
from data import DataFetcher;

FetcherLst = [];

def shutdownall():
    for fetcher in FetcherLst:
        if isinstance(fetcher, DataFetcher):
            fetcher.shutdown();
            
def runlayers(sizes):
    if not os.path.exists(preddir):
        os.mkdir(preddir);
    net_model = None;
    config = None;
    if not os.environ["CUDA_VISIBLE_DEVICES"]:
        net_model = net.build_model(net_name,net.PHASE.TEST,sizes);
        config = tf.ConfigProto(intra_op_parallelism_threads=4,device_count={'gpu':0});
    else:
        net_model = net.build_model(net_name,net.PHASE.TRAIN,sizes);
        config = tf.ConfigProto();
        config.gpu_options.allow_growth = True;
        config.allow_soft_placement = True;
        print "xx";
    ins = net_model[0];
    outs = net_model[1];
    opt_ops = net_model[2];
    sum_ops = net_model[3];
    test_fetcher = DataFetcher();
    test_fetcher.BATCH_SIZE = sizes[0];
    test_fetcher.PTS_DIM = sizes[1];
    test_fetcher.HID_NUM = sizes[2];
    test_fetcher.HEIGHT = sizes[3];
    test_fetcher.WIDTH = sizes[4];
    test_fetcher.Dir = util.listdir(testdir,".h5");
    test_fetcher.useMix = False;
    
    FetcherLst.append(test_fetcher);
    
    if len(net_model) > 4:
        test_fetcher.randfunc=net_model[4];
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer());
        saver = tf.train.Saver();
        ckpt = tf.train.get_checkpoint_state('%s/'%dumpdir);
        if ckpt and ckpt.model_checkpoint_path:  
            saver.restore(sess, ckpt.model_checkpoint_path);
        else:
            print "failed to restore model";
            return;
        stat = {};
        f = open(preddir+os.sep+"log.txt","w");
        layers = [];
        for op in tf.get_default_graph().get_operations():
            if ("Knn" in op.name or "rot" in op.name) and not "gradients" in op.name:
                print "op:",op.name;
                for tensor in op.inputs:
                    if tensor.shape and len(tensor.shape)==3 and int(tensor.shape[0])==BATCH_SIZE and int(tensor.shape[2])==3:
                        layers.append(tensor);
        try:
            test_fetcher.start();
            for cnt in range(len(test_fetcher.Dir)):
                x2D,x3D,yGT = test_fetcher.fetch();
                tag = test_fetcher.fetchTag();
                r2D = None;
                if len(ins) >= 5:
                    r2D_dim = int(ins[3].shape[1]);
                    r2D = np.random.normal(loc=0.0,scale=1.0,size=[BATCH_SIZE,r2D_dim]);
                #
                rgb = None;
                f_lst = None;
                if len(net_model) > 4:
                    rgb = util.sphere_to_YIQ( x3D );
                    tri_lst = util.triangulateSphere( x3D );
                    f_lst = util.getface(tri_lst);
                #
                yGTout = yGT.copy();
                yGT = yGT.reshape((-1,3));
                x3D = x3D.reshape((-1,3));
                yout=None;
                loss=None;
                if len(ins) < 5:
                    ylayers = sess.run(layers,feed_dict={ins[1]:x3D,ins[2]:x2D});
                else:
                    ylayers = sess.run(layers,feed_dict={ins[1]:x3D,ins[2]:x2D,ins[3]:r2D});
                fdir = preddir+os.sep+"pred_%s_%03d"%(tag,cnt);
                #
                if not os.path.exists(fdir):
                    os.mkdir(fdir);
                i = 0;
                for layer in layers:
                    lname = layer.name.replace(':','_');
                    if 1024 == ylayers[i].shape[1]:
                        if len(net_model) > 4:
                            util.write_to_obj(fdir+os.sep+"obj"+lname,ylayers[i],rgb,f_lst);
                        else:
                            util.write_to_obj(fdir+os.sep+"obj"+lname,ylayers[i]);
                    i += 1;
        finally:
            f.close();
            test_fetcher.shutdown();
    return;

if __name__ == "__main__":
    #some default value
    traindir="/data4T1/samhu/shapenet_split_complete/train";
    testdir="/data4T1/samhu/shapenet_split_complete/test";
    valdir="/data4T1/samhu/shapenet_split_complete/val";
    dumpdir="/data4T1/samhu/tf_dump/SL_Exp_04_train";
    preddir="/data4T1/samhu/tf_dump/predict";
    net_name="KPARAM_02";
    gpuid=1;
    for pt in sys.argv[1:]:
        if pt[:5]=="data=":
            datadir = pt[5:];
            traindir = datadir+"/train";
            testdir = datadir+"/test";
            valdir = datadir+"/val";
        elif pt[:5]=="dump=":
            dumpdir = pt[5:];
        elif pt[:5]=="pred=":
            preddir = pt[5:];
        elif pt[:4]=="gpu=":
            gpuid = int(pt[4:]);
        elif pt[:4]=="net=":
            net_name = pt[4:];
        else:
            cmd = pt;
    preddir += "/" + net_name;
    dumpdir += "/" + net_name;
    BATCH_SIZE=32;
    PTS_DIM=3;
    HID_NUM=32;
    HEIGHT=192;
    WIDTH=256;
    sizes=[BATCH_SIZE,PTS_DIM,HID_NUM,HEIGHT,WIDTH];
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%gpuid;
    try:
        if cmd=="run":
            runlayers(sizes);
        else:
            assert False,"input format wrong";
    finally:
        shutdownall();
        print "ended"