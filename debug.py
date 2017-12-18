import util;
from data import DataFetcher;
import net;
import os;

def debug_data():
    traindir="/data4T1/samhu/shapenet_split/train";
    fdir="./debug"
    BATCH_SIZE=16;
    PTS_DIM=3;
    HID_NUM=256;
    HEIGHT=192;
    WIDTH=256;
    sizes=[BATCH_SIZE,PTS_DIM,HID_NUM,HEIGHT,WIDTH];
    train_fetcher = DataFetcher();
    train_fetcher.BATCH_SIZE = sizes[0];
    train_fetcher.PTS_DIM = sizes[1];
    train_fetcher.HEIGHT = sizes[2];
    train_fetcher.WIDTH = sizes[3];
    train_fetcher.TrainDir = util.listdir(traindir);
    try:
        train_fetcher.start();
        x2D,x3D,yGT = train_fetcher.fetchTrain();
        rgb = util.sphere_to_YIQ( x3D );
        tri_lst = util.triangulateSphere( x3D );
        f_lst = util.getface(tri_lst);
        x3D_ellip = util.sphere_to_ellip(x3D,yGT);
        util.write_to_obj(fdir+os.sep+"x3D",x3D,rgb,f_lst);
        util.write_to_obj(fdir+os.sep+"x3D_ellip",x3D_ellip,rgb,f_lst);
        util.write_to_obj(fdir+os.sep+"yGT",yGT);
        util.write_to_img(fdir,x2D);
    finally:
        train_fetcher.shutdown();
    
def debug_net():
    BATCH_SIZE=32;
    PTS_DIM=3;
    HID_NUM=32;
    HEIGHT=192;
    WIDTH=256;
    net_name="KPARAM";
    sizes=[BATCH_SIZE,PTS_DIM,HID_NUM,HEIGHT,WIDTH];
    ret = net.build_model(net_name,net.PHASE.TRAIN,sizes);
    ins = ret[0];
    outs = ret[1];
    opt_ops = ret[2];
    sum_ops = ret[3];
if __name__ == "__main__":
    debug_net();
