# -*- coding: utf-8 -*-
"""
Spyder Editor

Use ALFW dataset

This is a temporary script file.
"""
import os
import tensorflow as tf 
import cv2
import time
import random

begin=time.time()
classes=['non-face','face']

face=os.listdir('E:\\friedhelm\\Data\\face\\')
others=os.listdir('E:\\friedhelm\\Data\\non-face\\')
random.shuffle(face)
random.shuffle(others)

kkk=0
print('train_start')
writer = tf.python_io.TFRecordWriter("E:\\friedhelm\\Data\\face_train_224.tfrecords")
for i in range(1,1000):
    if i%50==0:        
        print(i)
        print(time.time()-begin)
    for index, name in enumerate(classes):
        class_path='E:\\friedhelm\\Data\\'+name+'\\'
        if name=='face':
            docu_name=face
            p=list(range(50*(i-1),50*i))
        else:
            docu_name=others
            p=list(range(150*(i-1),150*i))
            
        for q in p:
            img_name=docu_name[q]
#        for img_name in docu_name[p]:
            img_path = class_path + img_name
#            print(img_path)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img,(224, 224))
            img_raw = img.tobytes()              #将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())  #序列化为字符串
            kkk+=1
writer.close()
print('train_end')
print(time.time()-begin)
print(kkk)

kkk=0
print('test_start')
writer = tf.python_io.TFRecordWriter("E:\\friedhelm\\Data\\face_test_224.tfrecords")
for i in range(1001,1200):
    if i%50==0:        
        print(i)
        print(time.time()-begin)
    for index, name in enumerate(classes):
        class_path='E:\\friedhelm\\Data\\'+name+'\\'
        if name=='face':
            docu_name=face
        else:
            docu_name=others
            
        for img_name in docu_name[50*(i-1):50*i]:
            img_path = class_path + img_name
#            print(img_path)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img,(224, 224))
            img_raw = img.tobytes()              #将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())  #序列化为字符串
            kkk+=1
writer.close()
print('test_end')
print(time.time()-begin)
print(kkk)

#7220