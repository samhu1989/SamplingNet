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

def train(sizes):
    if not os.path.exists(dumpdir):
        os.mkdir(dumpdir);
    net_model = net.build_model(net_name,net.PHASE.TRAIN,sizes);
    ins = net_model[0];
    outs = net_model[1];
    opt_ops = net_model[2];
    sum_ops = net_model[3];
        
    train_fetcher = DataFetcher();
    train_fetcher.BATCH_SIZE = sizes[0];
    train_fetcher.PTS_DIM = sizes[1];
    train_fetcher.HID_NUM = sizes[2];
    train_fetcher.HEIGHT = sizes[3];
    train_fetcher.WIDTH = sizes[4];
    train_fetcher.Dir = util.listdir(traindir);
    train_fetcher.shuffleDir();
    
    FetcherLst.append(train_fetcher);
    
    val_fetcher = DataFetcher();
    val_fetcher.BATCH_SIZE = sizes[0];
    val_fetcher.PTS_DIM = sizes[1];
    val_fetcher.HID_NUM = sizes[2];
    val_fetcher.HEIGHT = sizes[3];
    val_fetcher.WIDTH = sizes[4];
    val_fetcher.Dir = util.listdir(valdir);
    val_fetcher.shuffleDir();
    
    FetcherLst.append(val_fetcher);
    
    if len(net_model) > 4:
        train_fetcher.randfunc=net_model[4];
        val_fetcher.randfunc=net_model[4];
    
    config=tf.ConfigProto();
    config.gpu_options.allow_growth = True;
    config.allow_soft_placement = True;
    saver = tf.train.Saver();
    lrate = 3e-5;
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer());
        train_writer = tf.summary.FileWriter("%s/train"%(dumpdir),graph=sess.graph);
        valid_writer = tf.summary.FileWriter("%s/valid"%(dumpdir),graph=sess.graph)
        ckpt = tf.train.get_checkpoint_state('%s/'%dumpdir);
        if ckpt and ckpt.model_checkpoint_path:  
            saver.restore(sess, ckpt.model_checkpoint_path);
        try:
            train_fetcher.start();
            val_fetcher.start();
            lastEpoch = 0;
            while train_fetcher.EpochCnt < 100:
                x2D,x3D,yGT = train_fetcher.fetch();
                yGT = yGT.reshape((-1,3));
                x3D = x3D.reshape((-1,3));
                _,summary,step = sess.run([opt_ops[0],sum_ops[0],outs[-1]],feed_dict={ins[0]:yGT,ins[1]:x3D,ins[2]:x2D,ins[3]:lrate});
                train_writer.add_summary(summary,step);
                x2D,x3D,yGT = val_fetcher.fetch();
                yGT = yGT.reshape((-1,3));
                x3D = x3D.reshape((-1,3));
                summary,loss,step = sess.run([sum_ops[0],outs[-2],outs[-1]],feed_dict={ins[0]:yGT,ins[1]:x3D,ins[2]:x2D});
                valid_writer.add_summary(summary,step);
                if train_fetcher.EpochCnt >  ( lastEpoch + 10 ):
                    lastEpoch = train_fetcher.EpochCnt;
                    lrate *= 0.5;
                if step % 200 == 0:
                    saver.save(sess,'%s/'%dumpdir+"model_epoch%d.ckpt"%lastEpoch);
                epoch_len = len(train_fetcher.Dir);
                print "Epoch:",train_fetcher.EpochCnt,"step:",step,"/",epoch_len,"learning rate:",lrate;
        finally:
            train_fetcher.shutdown();
            val_fetcher.shutdown();
    return;

def test(sizes):
    os.environ["CUDA_VISIBLE_DEVICES"]="";
    if not os.path.exists(preddir):
        os.mkdir(preddir);
    net_model = net.build_model(net_name,net.PHASE.TEST,sizes);
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
    test_fetcher.Dir = util.listdir(testdir);
    test_fetcher.useMix = False;
    
    FetcherLst.append(test_fetcher);
    
    if len(net_model) > 4:
        test_fetcher.randfunc=net_model[4];
    
    config = tf.ConfigProto(intra_op_parallelism_threads=4,device_count={'gpu':0});
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
        try:
            test_fetcher.start();
            cnt = 0;
            while test_fetcher.EpochCnt < 2:
                x2D,x3D,yGT = test_fetcher.fetch();
                tag = test_fetcher.fetchTag();
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
                yout,loss = sess.run([outs[0],outs[1]],feed_dict={ins[0]:yGT,ins[1]:x3D,ins[2]:x2D});
                fdir = preddir+os.sep+"pred_%s_%03d"%(tag,cnt);
                #
                if not os.path.exists(fdir):
                    os.mkdir(fdir);
                if len(net_model) > 4:
                    util.write_to_obj(fdir+os.sep+"obj",yout,rgb,f_lst);
                else:
                    util.write_to_obj(fdir+os.sep+"obj",yout);
                util.write_to_obj(fdir+os.sep+"GTobj",yGTout);
                util.write_to_img(fdir,x2D);
                if tag in stat:
                    newcnt = stat[tag+"_cnt"] + 1;
                    stat[tag] = stat[tag]*stat[tag+"_cnt"]/newcnt + loss/newcnt;
                    stat[tag+"_cnt"] = newcnt;
                else:
                    stat[tag] = loss;
                    stat[tag+"_cnt"] = 1.0;
                cnt += 1;
                print "testing:tag=",tag,"loss=",loss,"mean loss of tag=",stat[tag];
            for (k,v) in stat.items():
                    print >>f,k,v;
        finally:
            f.close();
            test_fetcher.shutdown();
    return;

if __name__ == "__main__":
    #some default value
    traindir="/data4T1/samhu/shapenet_split/train";
    testdir="/data4T1/samhu/shapenet_split/test";
    valdir="/data4T1/samhu/shapenet_split/val";
    dumpdir="/data4T1/samhu/tf_dump/SL_Exp_04_train";
    preddir="/data4T1/samhu/tf_dump/predict";
    net_name="VPSGN";
    gpuid=1;
    for pt in sys.argv[1:]:
        if pt[:9]=="traindir=":
            traindir = pt[9:];
        elif pt[:8]=="testdir=":
            testdir = pt[8:];
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
        if cmd=="train":
            train(sizes);
        elif cmd=="test":
            test(sizes);
        elif cmd=="pretrain":
            pretrain(sizes);
        else:
            assert False,"input format wrong";
    finally:
        shutdownall();
        print "ended"