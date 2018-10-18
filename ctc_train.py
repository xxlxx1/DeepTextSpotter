# -*- coding: utf-8 -*-
import sys, os
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.insert(0, 'caffe/Release/install/python')
import caffe

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

import cv2
import math
import re
import random
import argparse
import numpy as np

from models import create_recognizer_solver
import utils
from utils import intersect, union, area, get_normalized_image, get_obox
from data import DataLoader

import matplotlib.pyplot as plt

#import vis
# import generate_codec_rev as gen
#from validation import validate    
image_no = 0

buckets = [54, 80, 124, 182, 272, 410, 614, 922, 1383, 2212]  
image_sizes = [[352, 352], [416, 416] ] #,[480, 480], [544, 544], [576, 576]]    
image_size = [160, 160]
it = 0
mean_loss = 0
mean_rec = 0
# bak is 256
data_batch = 384
data_total = 0
data_index = 0
f_list = []

valid_interval = 100
snapshot_interval = 4000

codec = u' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}~£ÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž'
codec_rev = {}
index = 4
for i in range(0, len(codec)):
  codec_rev[ord(codec[i])] = index
  index += 1

def process_batch( net , optim , args ):
  global it,data_batch,data_total,data_index,f_list,codec_rev,codec
  text = []
  W = []
  H = []
  bucket_images = {}
  dummy={}
  net_ctc = net.net
  #print( len(codec))
  #1 Read images and gts ( circular buffer form for small samples)   
  for img_ind in range( min(data_batch,data_total) ):
    circ_ind = (data_index + img_ind ) % data_total
    
    img_path = os.path.join( args.data_dir, f_list[circ_ind].replace('../','') )
    #print('img_ind=%d,img_path=%s' %(data_index,img_path))
    img=cv2.imread( img_path.strip() )
    if img.shape[2]==3:
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img == None:
      print("im read error")
      print("img_path:%s" %img_path)
    W = img.shape[1]
    H = img.shape[0]
    with open( img_path.strip().replace('.jpg','.txt')) as f:
      item = f.readlines()
    anns = item[0].split()
    text = anns[6].decode('utf8')
    data_index+=1    
    #2 Adjust image sizes to fixed height and variable width
    width_scale = 32.0 / H
    width = W * width_scale
           
    best_diff = width
    bestb = 0
    for b in range(0, len(buckets)):
      if best_diff > abs(width * 1.3 - buckets[b]):
        best_diff = abs(width * 1.3 - buckets[b])
        bestb = b
        
    scaled = cv2.resize(img, (buckets[bestb], 32))  
    scaled = np.asarray(scaled, dtype=np.float)
    delta = scaled.max() - scaled.min()
    #print('scaled.max=%d scaled.min=%d delta=%d mean=%d' %(scaled.max(),scaled.min(),delta, scaled.mean()))
    scaled = (scaled) / (delta / 2.0)
    scaled -= scaled.mean()
    #print( 'scaled')
    #print(scaled.shape )
    if not bucket_images.has_key(bestb):
      bucket_images[bestb] = {}
      bucket_images[bestb]['img'] = []  
      bucket_images[bestb]['sizes'] = []    
      bucket_images[bestb]['txt'] = []
      bucket_images[bestb]['gt_enc'] = []
      dummy[bestb] = 1
    gt_labels = []
    txt_enc = ''
    for k in range( len(text) ):
      t_unicode = text[k]
      if t_unicode > 0:
        if codec_rev.has_key( t_unicode ):
          gt_labels.append( codec_rev[ t_unicode ] )
        else:
          gt_labels.append( 3 )
      else:
        gt_labels.append( 0 )
    
    if scaled.ndim==3:
      print( scaled.shape )
      scaled = cv2.cvtColor(scaled, cv2.COLOR_RGB2GRAY)
    if args.debug:
      cv2.imshow('scaled', scaled)
    bucket_images[bestb]['sizes'].append(len(gt_labels))
    bucket_images[bestb]['gt_enc'].append(gt_labels)
    bucket_images[bestb]['txt'].append(text)
    bucket_images[bestb]['img'].append(scaled)
  data_index = data_index % data_total 
  
  #3 Transfer the data into the net 
  for bucket in bucket_images.keys():
    #print(bucket)  
    imtf = np.asarray(bucket_images[bucket]['img'], dtype=np.float)
    #print('imtf.shape')
    #print( imtf.shape )
    imtf = np.reshape(imtf, (imtf.shape[0], -1, imtf.shape[1], imtf.shape[2]))        
    #print('imtf reshape')
    #print( 'imtf.shape[0]=%d imtf.shape[1]=%d imtf.shape[2]=%d imtf.shape[3]=%d' %(imtf.shape[0],imtf.shape[1],imtf.shape[2],imtf.shape[3]) )
    net_ctc.blobs['data'].reshape(imtf.shape[0],imtf.shape[1],imtf.shape[2], imtf.shape[3]) 
    net_ctc.blobs['data'].data[...] = imtf
    
    labels = bucket_images[bucket]['gt_enc']
    txt = bucket_images[bucket]['txt']
    # indentical length needed     
    max_len = 0
    for l in range(0, len(labels)):
      max_len = max(max_len, len(labels[l]))
    for l in range(0, len(labels)):
      while len(labels[l]) <  max_len:
        labels[l].append(0)
      
    
    labels = np.asarray(labels, np.float)
    
    net_ctc.blobs['label'].reshape(labels.shape[0], labels.shape[1])
    
    net_ctc.blobs['label'].data[...] = labels    
    #4 Compute forward-backward
    #optim.step(1)
    net.step(1)
    it +=1
    #5 If loss is large, print it
    #if net_ctc.blobs['loss'].data[...] > 10:
    sf = net_ctc.blobs['transpose'].data[...]
    #print( 'sf.shape' )
    #print(sf.shape)
    labels2 = sf.argmax(3)
    out = utils.print_seq(labels2[:,0, :])
    print(u'{0} <--> {1}'.format(out, txt[0])  )

    if it%snapshot_interval == 0:
      #optim.snapshot()
      net.snapshot()
      print ( 'it is %d, and snapshot_interval is %d' %(it,snapshot_interval) )
      print( 'snapshot saved')


def train_dir( net, optim,args):  
  caffe.set_mode_gpu()
  while True:
    process_batch(net,optim, args)


parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', default='/data/xlxia/code/icdar_2017')
parser.add_argument('-train_list', default='/data/xlxia/code/icdar_2017/train.txt')
parser.add_argument('-debug', type=int, default=0)

args = parser.parse_args()

with open(args.train_list,'r') as f:
  f_list = f.readlines()

data_total = len( f_list ) 

net_ctc_sgd = create_recognizer_solver(args)
#net_ctc_sgd.net.copy_from('models/model.caffemodel')
net = net_ctc_sgd
train_dir( net ,net_ctc_sgd,args)




