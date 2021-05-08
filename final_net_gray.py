########################################################
# flownet+brightness consistency loss fn + image warping
######################################################## 
import os
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tf_slim as slim #tensorflow.contrib.slim as slim
import random
import numpy as np
import shutil
import struct
import time
#import tensorflow_addons as tfa
import matplotlib.pyplot as plt 
from flo2img import flow_to_image
from scipy import ndimage
#import flow_vis
#from PIL import Image

dir0 = 'train/'
net_name = 'final_net_modified_warp/'
dir_restore = 'restore/'
dir_data = '/content/drive/MyDrive/First300' 
lr_base = 1e-4              # initial learning rate
epoch_lr_decay = 500        # every # epoch, lr will decay 0.1
epoch_max = 2             # max epoch
max_to_keep = 1             # number of model to save
batch_size =20         # bs
train_pairs_number = 100  # number of train samples
val_iter = 1             # validation batch
use_gpu_1 = True
W, H = 512,384   #256,192
epsilon = 10^(-7)

val_pairs_number = batch_size * val_iter
iter_per_epoch = train_pairs_number // batch_size
epoch_save = epoch_max // max_to_keep
########################################
dir_models = 'model/' + net_name
dir_logs = 'log/' + net_name
dir_model = dir_models + dir0
dir_log_train = dir_logs + dir0 + '_train'
dir_log_test = dir_logs + dir0 + '_test'
if not os.path.exists(dir_models):
    os.makedirs(dir_models)
if not os.path.exists(dir_logs):
    os.makedirs(dir_logs)
if os.path.exists(dir_model):
    shutil.rmtree(dir_model)
if os.path.exists(dir_log_train):
    shutil.rmtree(dir_log_train)
if os.path.exists(dir_log_test):
    shutil.rmtree(dir_log_test)

os.makedirs(dir_model)
os.makedirs(dir_log_train)
os.makedirs(dir_log_test)
########################################
kernelx = np.array([[0,0.25,0.25],
                     [0,-0.25,-0.25],
                     [0,0,0]], dtype = 'float32')

kernely = np.array([[0,-0.25,0.25],
                     [0,-0.25,0.25],
                     [0,0,0]], dtype = 'float32')

kernelt = np.array([[0,0.25,0.25],
                     [0,0.25,0.25],
                     [0,0,0]], dtype = 'float32')


def initx(shape, dtype=None):
    return kernelx[..., None, None]

def inity(shape, dtype=None):
    return kernely[..., None, None]

def initt(shape, dtype=None):
    return kernelt[..., None, None]


########################################
def remove_file(directory_list):
    if '.directory' in directory_list:
        directory_list.remove('.directory')
    return directory_list


def load_data():
    img1_list_t = []
    img2_list_t = []
    flow_list_t = []
    img1_list_v = []
    img2_list_v = []
    flow_list_v = []
    namelist = remove_file(os.listdir(dir_data))
    namelist.sort()
    print(namelist)
    for i in range(train_pairs_number+val_pairs_number):
        if i < train_pairs_number:
            flow_list_t.append(dir_data +"/"+ namelist[3*i])
            img1_list_t.append(dir_data + "/"+namelist[3*i+1])
            img2_list_t.append(dir_data + "/"+namelist[3*i+2])
        else:
            flow_list_v.append(dir_data +"/"+ namelist[3*i])
            img1_list_v.append(dir_data + "/"+namelist[3*i+1])
            img2_list_v.append(dir_data + "/"+namelist[3*i+2])

    #print(len(img1_list_t))
    assert len(img1_list_t) == len(img2_list_t)
    assert len(img1_list_t) == len(flow_list_t)
    assert len(img1_list_v) == len(img2_list_v)
    assert len(img1_list_v) == len(flow_list_v)
    return img1_list_t, img2_list_t, flow_list_t, img1_list_v, img2_list_v, flow_list_v

def im_warp(img, predict):
    img2 = np.zeros((batch_size,H,W,1))
    prev = np.zeros((H,W,1))
    for batch in range(batch_size):
        #print(batch, batch_size, predict.shape)
        flow = predict[batch,:,:,:]
        plt.imshow(flow_to_image(flow))
        plt.show()
         #flow_ans = flow.shape[1]
         #print(type(flow_ans))
        curImg = img[batch,:,:,:]
         
         #print(curImg[:,:,0].max())
        h, w = flow.shape[:2]
         #h = H
         #w = W
        flow = (np.float32(flow))
         # # #flow1 = flow.np
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        # plt.imshow(curImg, 'gray')
        # plt.show()
         # curImg2 = rgb2gray(curImg)
         #print(flow.dtype)
        prevImg = cv2.remap(curImg, flow, None, cv2.INTER_LINEAR)
        # plt.imshow(prevImg, 'gray')
        # plt.show()
        prev[:,:,0] = np.array(prevImg)
        img2[batch,:,:,:] = prev         #print(img2.shape)
    return img2

class Data(object):
    def __init__(self, list1, list2, list3, bs=batch_size, shuffle=True, minus_mean=True):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        self.bs = bs #2
        self.index = 0  
        self.number = len(self.list1) #2
        self.index_total = list(range(self.number)) #[0,1]
        self.shuffle = shuffle
        self.minus_mean = minus_mean
        if self.shuffle: random.shuffle(self.index_total)

    def read_flow(self, name): # used to read the flow of code
        f = open(name, "rb")
        data = f.read()
        f.close()
        width = struct.unpack('@i', data[4:8])[0]
        height = struct.unpack('@i', data[8:12])[0]
        flowdata = np.zeros((height, width, 2))
        for i in range(width*height):
            data_u = struct.unpack('@f', data[12+8*i:16+8*i])[0]
            data_v = struct.unpack('@f', data[16+8*i:20+8*i])[0]
            n = int(i / width)
            k = np.mod(i, width)
            flowdata[n, k, :] = [data_u, data_v]
        return flowdata

    def next_batch(self):
        start = self.index #0
        self.index += self.bs #2
        if self.index > self.number:
            if self.shuffle: random.shuffle(self.index_total)
            self.index = 0
            start = self.index
            self.index += self.bs
        end = self.index
        
        img1 = np.zeros((H,W,1))
        img2 = np.zeros((H,W,1))
        flow = np.zeros((H,W,2))
        
        img1_batch = np.zeros((batch_size, H,W,1))
        img2_batch = np.zeros((batch_size, H,W,1))
        flow_batch = np.zeros((batch_size, H,W,2))
        for i in range(start, end):
            #print(start, end, i, self.index_total[i])
            ii = i-start
            img = cv2.resize((cv2.imread(self.list1[self.index_total[i]], cv2.IMREAD_GRAYSCALE).astype(np.float32)),(W, H))
            #img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # plt.imshow(img,'gray')
            # plt.show()
            img1[:,:,0] = np.array(img)
            img1_batch[ii,:,:,:] = img1
            img = cv2.resize((cv2.imread(self.list2[self.index_total[i]], cv2.IMREAD_GRAYSCALE).astype(np.float32)),(W, H))
            #img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img2[:,:,0] = np.array(img)
            img2_batch[ii,:,:,:] = img2
           
            flow1 = self.read_flow(self.list3[self.index_total[i]])
            flow[:,:,0] = cv2.resize(flow1[:,:,0], (W, H))
            flow[:,:,1] = cv2.resize(flow1[:,:,1], (W, H))
            flow_batch[ii,:,:,:] = np.array(flow)

        return img1_batch, img2_batch, flow_batch

   
    
class Net(object):
    def __init__(self, use_gpu_1=True):
        #print('Hi')
        self.x1 = tf.placeholder(tf.float32, [None, H, W, 1], name='x1')  # image1
        self.x2 = tf.placeholder(tf.float32, [None, H, W, 1], name='x2')  # image2
        self.x3 = tf.placeholder(tf.float32, [None, H, W, 2], name='x3')  # label      
        self.lr = tf.placeholder(tf.float32, [], name='lr')  # lr
        #for loop in range(1,4):
        with tf.variable_scope('conv'):
            
            concat1 = tf.concat([self.x1, self.x2] ,3)
            conv1 = slim.conv2d(concat1, 96, [7, 7], 1, scope='conv1')
            conv2 = slim.conv2d(conv1, 128, [5, 5], 2, scope='conv2')
            conv2_1 = slim.conv2d(conv2, 128, [3, 3], 1, scope='conv2_1')
            conv3 = slim.conv2d(conv2_1, 128, [5, 5], 2, scope='conv3')
            conv3_1 = slim.conv2d(conv3, 128, [3, 3], 1, scope='conv3_1')
            conv4 = slim.conv2d(conv3_1, 256, [3, 3], 1, scope='conv4')
            conv4_1 = slim.conv2d(conv4, 512, [3, 3], 2, scope='conv4_1')
            conv5 = slim.conv2d(conv4_1, 512, [1, 1], 2, scope='conv5')
            conv5_10 = slim.conv2d(conv5, 256, [3, 3], 1, scope='conv5_10')
            conv5_1 = slim.conv2d(conv5_10, 256, [3, 3], 1, scope='conv5_1')
            conv6 = slim.conv2d(conv5_1, 64, [5, 5], 2, scope='conv6')
            #conv6_1 = slim.conv2d(conv6, 1024, [3, 3], 1, scope='conv6_1')
            self.predict6 = slim.conv2d(conv6, 2, [3, 3], 1, activation_fn=None, scope='pred6')

        with tf.variable_scope('deconv'):
            # 24 * 32 flow
            deconv5 = slim.conv2d_transpose(conv6, 256, [4, 4], 2, scope='deconv5')
            deconvflow6 = slim.conv2d_transpose(self.predict6, 2, [4, 4], 2, 'SAME', scope='deconvflow6')
            concat5 = tf.concat([conv5_1, deconv5, deconvflow6],3, name='concat5')
            self.predict5 = slim.conv2d(concat5, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict5')
            # 48 * 64 flow
            deconv4 = slim.conv2d_transpose(concat5, 128, [4, 4], 2, 'SAME',  scope='deconv4')
            deconvflow5 = slim.conv2d_transpose(self.predict5, 2, [4, 4], 2, 'SAME', scope='deconvflow5')
            concat4 = tf.concat([conv4_1, deconv4, deconvflow5], 3,name='concat4')
            self.predict4 = slim.conv2d(concat4, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict4')
            # 96 * 128 flow
            deconv3 = slim.conv2d_transpose(concat4, 128, [4, 4], 2, 'SAME', scope='deconv3')
            deconvflow4 = slim.conv2d_transpose(self.predict4, 2, [4, 4], 2, 'SAME',scope='deconvflow4')
            concat3 = tf.concat([conv3_1, deconv3, deconvflow4],3, name='concat3')
            self.predict3 = slim.conv2d(concat3, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict3')
            # 192 * 256 flow
            deconv2 = slim.conv2d_transpose(concat3, 64, [4, 4], 2, 'SAME', scope='deconv2')
            deconvflow3 = slim.conv2d_transpose(self.predict3, 2, [4, 4], 2, 'SAME', scope='deconvflow3')
            concat2 = tf.concat([conv2, deconv2, deconvflow3], 3, name='concat2')
            self.predict2 = slim.conv2d(concat2, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict2')
            # 384 * 512 flow
            deconv1 = slim.conv2d_transpose(concat2, 32, [4, 4], 2, 'SAME', scope='deconv1')
            deconvflow2 = slim.conv2d_transpose(self.predict2, 2, [4, 4], 2, 'SAME', scope='deconvflow2')
            concat1 = tf.concat([conv1, deconv1, deconvflow2], 3, name='concat1')
            self.predict1 = slim.conv2d(concat1, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict1')

             
        self.tvars = tf.trainable_variables()  # turn on if you want to check the variables
        # self.variables_names = [v.name for v in self.tvars]

        with tf.variable_scope('loss'):
               weight = [1.0/2, 1.0/4, 1.0/8, 1.0/16, 1.0/32, 1.0/32]
               i1 = tf.image.resize_images(self.x1, [int(H/32), int(W/32)])
               i2 = tf.image.resize_images(self.x2, [int(H/32), int(W/32)])
               loss6 = weight[5] * self.charbonnier(i1,i2, self.predict6)
               i1 = tf.image.resize_images(self.x1, [int(H/16), int(W/16)])
               i2 = tf.image.resize_images(self.x2, [int(H/16), int(W/16)])
               loss5 = weight[4] * self.charbonnier(i1,i2, self.predict5)
               i1 = tf.image.resize_images(self.x1, [int(H/8), int(W/8)])
               i2 = tf.image.resize_images(self.x2, [int(H/8), int(W/8)])
               loss4 = weight[3] * self.charbonnier(i1,i2, self.predict4)
               i1 = tf.image.resize_images(self.x1, [int(H/4), int(W/4)])
               i2 = tf.image.resize_images(self.x2, [int(H/4), int(W/4)])
               loss3 = weight[2] * self.charbonnier(i1,i2, self.predict3)
               i1 = tf.image.resize_images(self.x1, [int(H/2), int(W/2)])
               i2 = tf.image.resize_images(self.x2, [int(H/2), int(W/2)])
               loss2 = weight[1] * self.charbonnier(i1,i2, self.predict2)
               i1 = self.x1
               i2 = self.x2
               loss1 = weight[0] * self.charbonnier(i1,i2, self.predict1)
            
               self.loss = tf.add_n([loss6,loss5, loss4, loss3, loss2, loss1])
               tf.summary.scalar('loss6', loss6)
               tf.summary.scalar('loss5', loss5)
               tf.summary.scalar('loss4', loss4)
               tf.summary.scalar('loss3', loss3)
               tf.summary.scalar('loss2', loss2)
               tf.summary.scalar('loss1', loss1)
               tf.summary.scalar('loss', self.loss)
               self.merged = tf.summary.merge_all() #tf.summary.all_v2_summary_ops()
            
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = slim.learning.create_train_op(self.loss, optimizer)

        # gpu configuration
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        if use_gpu_1:
            self.tf_config.gpu_options.visible_device_list = '1'

        self.init_all = tf.initialize_all_variables()

    def mean_loss(self, gt, predict):
        loss = tf.reduce_mean(tf.abs(gt-predict))
        return loss
    
    def charbonnier(self, i1,i2,predict):
        Ix = slim.conv2d((i1 + i2), 1 , [3, 3], weights_initializer = initx, trainable = False)
        Iy = slim.conv2d((i1 + i2), 1 , [3, 3], weights_initializer = inity, trainable = False)
        It = slim.conv2d((i2 - i1), 1 , [3, 3], weights_initializer = initt, trainable = False)
        
        
        u,v = tf.split(predict, 2, axis=3)
        loss = tf.reduce_mean(tf.abs(Ix*u + Iy*v + It))
        #loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(Ix*u + Iy*v + It)+epsilon)))
        return loss
        
def main():
    # data preparation
    list1_t, list2_t, list3_t, list1_v, list2_v, list3_v = load_data()
    dataset_t = Data( list1_t, list2_t,list3_t, shuffle=True, minus_mean=False)
    dataset_v = Data( list1_v, list2_v,list3_v, shuffle=True, minus_mean=False)
    x1_v = []
    x2_v = []
    x3_v = []
    for j in range(val_iter):
        x1_b, x2_b, x3_b = dataset_v.next_batch()
        x1_v.append(x1_b)
        x2_v.append(x2_b)
        x3_v.append(x3_b)

    model = Net(use_gpu_1=use_gpu_1)
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    with tf.Session(config=model.tf_config) as sess:
        sess.run(model.init_all)
        #saver.restore(sess, dir_restore)
        writer_train = tf.summary.FileWriter(dir_log_train, sess.graph)
        writer_val = tf.summary.FileWriter(dir_log_test, sess.graph)
        for epoch in range(epoch_max):
            lr_decay = 0.01 ** (epoch / epoch_lr_decay)
            lr = lr_base * lr_decay
            for iteration in range(iter_per_epoch):
                time_start = time.time()
                global_iter = epoch * iter_per_epoch + iteration
                #predict = tf.zeros([batch_size, H, W, 2], dtype = tf.float32)
                predict = np.zeros(((batch_size, H,W, 2)))
                x1_t, x2_t, x3_t = dataset_t.next_batch()
                for loop in range(4):
                    print(loop)
                    
                    feed_dict = {model.x1: x1_t, model.x2: x2_t, model.x3: x3_t, model.lr: lr}
                    #print('feeding successful')
                    _, merged_out_t, loss_out_t, predictf = sess.run([model.train_op, model.merged, model.loss, model.predict1], feed_dict)
                    #predictf = sess.run(model.predict1, feed_dict)
                    #print(type(predictf))
                    predict =predict + predictf
                    predict = ndimage.median_filter(predict, size=5)
                    x2_t = im_warp(x2_t, predict) 
                    
                    
               
                
                writer_train.add_summary(merged_out_t, global_step = global_iter + 1)
                hour_per_epoch = iter_per_epoch * ((time.time() - time_start) / 3600)
                print('%.4f h/epoch, epoch %03d/%03d, iter %04d/%04d, lr %.5f, loss: %.5f' %
                      (hour_per_epoch, epoch + 1, epoch_max, iteration + 1, iter_per_epoch, lr, loss_out_t))
                #predict_out = sess.run(model.predict1, feed_dict)
                # plt.imshow(flow_to_image(predict[0,:,:,:]))
                # plt.show()
                
                
                
                if not (iteration + 1) % (train_pairs_number// batch_size):
                    feed_dict_v = {model.x1: x1_v[0], model.x2: x2_v[0], model.x3: x3_v[0]}
                    merged_out_v, loss_out_v = sess.run([model.merged, model.loss], feed_dict_v)
                    print('****val loss****: %.5f' % loss_out_v)
                    writer_val.add_summary(merged_out_v, global_iter + 1)

            # save
            if not (epoch + 1) % epoch_save:
                saver.save(sess, (dir_model + '/model'), global_step=epoch+1)


if __name__ == "__main__":
      tf.app.run()
