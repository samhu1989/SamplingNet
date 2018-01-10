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
    train_fetcher.Dir = util.listdir(traindir,".h5");
    train_fetcher.shuffleDir();
    
    FetcherLst.append(train_fetcher);
    
    val_fetcher = DataFetcher();
    val_fetcher.BATCH_SIZE = sizes[0];
    val_fetcher.PTS_DIM = sizes[1];
    val_fetcher.HID_NUM = sizes[2];
    val_fetcher.HEIGHT = sizes[3];
    val_fetcher.WIDTH = sizes[4];
    val_fetcher.Dir = util.listdir(valdir,".h5");
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
                out = train_fetcher.fetch();
                x2D = out[0];
                x3D = out[1];
                yGT = out[-1]; 
                GT_PTS_NUM = int(yGT.shape[1]);
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
                print "Epoch:",train_fetcher.EpochCnt,"GT_PTS_NUM",GT_PTS_NUM,"step:",step,"/",epoch_len,"learning rate:",lrate;
        finally:
            train_fetcher.shutdown();
            val_fetcher.shutdown();
    return;

def train_minofn(sizes):
    min_of_N = 2;
    if not os.path.exists(dumpdir):
        os.mkdir(dumpdir);
    net_model = net.build_model(net_name,net.PHASE.TRAIN,sizes);
    ins = net_model[0];
    outs = net_model[1];
    opt_ops = net_model[2];
    sum_ops = net_model[3];
    r2D_dim = int(ins[3].shape[1]);
        
    train_fetcher = DataFetcher();
    train_fetcher.BATCH_SIZE = sizes[0];
    train_fetcher.PTS_DIM = sizes[1];
    train_fetcher.HID_NUM = sizes[2];
    train_fetcher.HEIGHT = sizes[3];
    train_fetcher.WIDTH = sizes[4];
    train_fetcher.Dir = util.listdir(traindir,".h5");
    train_fetcher.shuffleDir();
    
    FetcherLst.append(train_fetcher);
    
    val_fetcher = DataFetcher();
    val_fetcher.BATCH_SIZE = sizes[0];
    val_fetcher.PTS_DIM = sizes[1];
    val_fetcher.HID_NUM = sizes[2];
    val_fetcher.HEIGHT = sizes[3];
    val_fetcher.WIDTH = sizes[4];
    val_fetcher.Dir = util.listdir(valdir,".h5");
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
            saver.restore(sess,ckpt.model_checkpoint_path);
        try:
            train_fetcher.start();
            val_fetcher.start();
            lastEpoch = 0;
            while train_fetcher.EpochCnt < 100:
                out = train_fetcher.fetch();
                x2D = out[0];
                x3D = out[1];
                yGT = out[-1]; 
                GT_PTS_NUM = int(yGT.shape[1]);
                yGT = yGT.reshape((-1,3));
                x3D = x3D.reshape((-1,3));
                minr2D = np.random.normal(loc=0.0,scale=1.0,size=[BATCH_SIZE,r2D_dim]);
                mindists = np.zeros([BATCH_SIZE],np.float32);
                for min_of_i in range(min_of_N):
                    r2D = np.random.normal(loc=0.0,scale=1.0,size=[BATCH_SIZE,r2D_dim]);
                    dists = sess.run(outs[1],feed_dict={ins[0]:yGT,ins[1]:x3D,ins[2]:x2D,ins[3]:r2D});
                    for i in range(mindists.size):
                        if mindists[i] == 0 or mindists[i] > dists[i]:
                            mindists[i] = dists[i];
                            minr2D[i,...] = r2D[i,...];
                _,summary,step = sess.run([opt_ops[0],sum_ops[0],outs[-1]],feed_dict={ins[0]:yGT,ins[1]:x3D,ins[2]:x2D,ins[3]:minr2D,ins[4]:lrate});
                train_writer.add_summary(summary,step);
                x2D,x3D,yGT = val_fetcher.fetch();
                r2D = np.random.normal(loc=0.0,scale=1.0,size=[BATCH_SIZE,r2D_dim]);
                yGT = yGT.reshape((-1,3));
                x3D = x3D.reshape((-1,3));
                summary,loss,step = sess.run([sum_ops[0],outs[-2],outs[-1]],feed_dict={ins[0]:yGT,ins[1]:x3D,ins[2]:x2D,ins[3]:r2D});
                valid_writer.add_summary(summary,step);
                if train_fetcher.EpochCnt >  ( lastEpoch + 10 ):
                    lastEpoch = train_fetcher.EpochCnt;
                    lrate *= 0.5;
                if step % 200 == 0:
                    saver.save(sess,'%s/'%dumpdir+"model_epoch%d.ckpt"%lastEpoch);
                epoch_len = len(train_fetcher.Dir);
                print "Epoch:",train_fetcher.EpochCnt,"GT_PTS_NUM",GT_PTS_NUM,"step:",step,"/",epoch_len,"learning rate:",lrate;
        finally:
            train_fetcher.shutdown();
            val_fetcher.shutdown();
    return;

def test(sizes):
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
                    yout,loss = sess.run([outs[0],outs[1]],feed_dict={ins[0]:yGT,ins[1]:x3D,ins[2]:x2D});
                else:
                    yout,loss = sess.run([outs[0],outs[2]],feed_dict={ins[0]:yGT,ins[1]:x3D,ins[2]:x2D,ins[3]:r2D});
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
                print "testing:tag=",tag,"loss=",loss,"mean loss of tag=",stat[tag];
                #generating dense result
                if len(net_model) > 4:
                    ptsnum = int(yGTout.shape[1]);
                    x3Ddense = np.zeros([BATCH_SIZE,ptsnum*10,3],np.float32);
                    ydense = np.zeros([BATCH_SIZE,ptsnum*10,3],np.float32);
                    for i in range(10):
                        x3D = util.rand_n_sphere(BATCH_SIZE,ptsnum);
                        x3Ddense[:,i*ptsnum:(i+1)*ptsnum,0:3] = x3D;
                        x3D = x3D.reshape((-1,3));
                        if len(ins) < 5:
                            yout = sess.run(outs[0],feed_dict={ins[1]:x3D,ins[2]:x2D});
                        else:
                            yout = sess.run(outs[0],feed_dict={ins[1]:x3D,ins[2]:x2D,ins[3]:r2D});
                        ydense[:,i*ptsnum:(i+1)*ptsnum,0:3] = yout;
                    rgbdense = util.sphere_to_YIQ( x3Ddense );
                    tri_lstdense = util.triangulateSphere( x3Ddense );
                    f_lstdense = util.getface(tri_lstdense);
                    util.write_to_obj(fdir+os.sep+"objdense",ydense,rgbdense,f_lstdense);
                        
            for (k,v) in stat.items():
                    print >>f,k,v;
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
    net_name="VPSGN";
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
        if cmd=="train":
            train(sizes);
        elif cmd=="cputest":
            os.environ["CUDA_VISIBLE_DEVICES"]="";
            test(sizes);
        elif cmd=="test":
            test(sizes);
        elif cmd=="trainrand":
            train_minofn(sizes);
        else:
            assert False,"input format wrong";
    finally:
        shutdownall();
        print "ended"