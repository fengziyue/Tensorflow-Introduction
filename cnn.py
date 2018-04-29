from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import re
import sys
import tarfile
import numpy as np
from six.moves import urllib
import tensorflow as tf
import os
from datetime import datetime
import os.path
import time
from six.moves import xrange 

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 182079

def inputs():

  batch_size=64
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  
  read_input=tf.placeholder(tf.uint8)
  read_input = read_images()
  reshaped_image=tf.placeholder(tf.float32)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
  min_fraction_of_examples_in_queue)

  return generate_batch(reshaped_image, read_input.label,
  min_queue_examples, 64)
def read_images():
  class CIFAR10Record(object):
  pass
  result = CIFAR10Record()

  result.height = 256
  result.width = 320
  result.depth = 3

  with open('train.txt') as fid:
  content = fid.read()
  content = content.split('\n')
  content = content[:-1]          #消除换行符\n的影响

  valuequeue = tf.train.string_input_producer(content,shuffle=True)
  value = valuequeue.dequeue()

  dir, label1,label2,label3,label4 = tf.decode_csv(records=value,
   record_defaults=[['string'], [''],[''],[''],['']], field_delim=" ")

  label1 = tf.string_to_number(label1, tf.float32)
  label2 = tf.string_to_number(label2, tf.float32)
  label3 = tf.string_to_number(label3, tf.float32)
  label4 = tf.string_to_number(label4, tf.float32)

  result.label=tf.stack([label1,label2,label3,label4])
  imagecontent = tf.read_file(dir)
  image = tf.image.decode_jpeg(imagecontent, channels=3)
  result.uint8image=image
  return result

def generate_batch(image, label, min_queue_examples,
  batch_size):
  
  num_preprocess_threads = 8     #多线程读取

  images=tf.placeholder(tf.float32)
  label_batch=tf.placeholder(tf.float32)
  images, label_batch = tf.train.batch(
  [image, label],
  batch_size=64,
shapes=([256,320,3],[4]),
  num_threads=num_preprocess_threads,
  capacity=50000)
  
  return images, tf.reshape(label_batch, [batch_size,4])

def inference(images):
  # conv1
  with tf.variable_scope('conv1') as scope:
  norm1=tf.placeholder("float",shape=[None,256,320,3])
  conv1=tf.placeholder("float")
  conv=tf.placeholder("float")
  bias=tf.placeholder("float")
  norm1 = tf.nn.lrn(images, 4, bias=255.0, alpha=0.0, beta=1.0,
                    name='norm1')
  norm1=norm1-0.5
  tf.summary.histogram('norm1' + '/activations', norm1) 
  kernel = tf.get_variable('weights',
  shape=[5, 5, 3, 24],
  initializer=tf.contrib.layers.xavier_initializer())
  conv = tf.nn.conv2d(norm1, kernel, [1, 2, 2, 1], padding='VALID')
  biases = tf.get_variable('biases', shape=[24],
  initializer=tf.constant_initializer(0.1))
  weight=tf.reduce_sum(kernel)/(5*5*3*24)
  biases_ave=tf.reduce_sum(biases)/24
  bias = tf.nn.bias_add(conv, biases)
  conv1 = tf.nn.relu(bias)
  tf.summary.scalar('conv1' + '/weight', weight)
  tf.summary.scalar('conv1' + '/biases', biases_ave)
  tf.summary.histogram('conv1' + '/activations', conv1)
  
  # conv2
  with tf.variable_scope('conv2') as scope:
  conv2=tf.placeholder("float")
  conv=tf.placeholder("float")
  bias=tf.placeholder("float")
  kernel = tf.get_variable('weights',
  shape=[5, 5, 24, 36],
  initializer=tf.contrib.layers.xavier_initializer())
  conv = tf.nn.conv2d(conv1, kernel, [1, 2, 2, 1], padding='VALID')
  biases = tf.get_variable('biases', shape=[36], initializer=tf.constant_initializer(0.1))
  weight=tf.reduce_sum(kernel)/(5*5*36*24)
  biases_ave=tf.reduce_sum(biases)/36
  bias = tf.nn.bias_add(conv, biases)
  conv2 = tf.nn.relu(bias)
  tf.summary.scalar('conv2' + '/weight', weight)
  tf.summary.scalar('conv2' + '/biases', biases_ave)
  tf.summary.histogram('conv2' + '/activations', conv2)
  
  # conv3
  with tf.variable_scope('conv3') as scope:
  conv3=tf.placeholder("float")
  conv=tf.placeholder("float")
  bias=tf.placeholder("float")
  kernel = tf.get_variable('weights',
  shape=[5, 5, 36, 48],
  initializer=tf.contrib.layers.xavier_initializer())
  conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='VALID')
  biases = tf.get_variable('biases', shape=[48], initializer=tf.constant_initializer(0.1))
  weight=tf.reduce_sum(kernel)/(5*5*36*48)
  biases_ave=tf.reduce_sum(biases)/48
  bias = tf.nn.bias_add(conv, biases)
  conv3 = tf.nn.relu(bias)
  tf.summary.scalar('conv3' + '/weight', weight)
  tf.summary.scalar('conv3' + '/biases', biases_ave)
  tf.summary.histogram('conv3' + '/activations', conv3)
  
  # conv4
  with tf.variable_scope('conv4') as scope:
  conv4=tf.placeholder("float")
  conv=tf.placeholder("float")
  bias=tf.placeholder("float")
  kernel = tf.get_variable('weights',
  shape=[3, 3, 48, 64],
  initializer=tf.contrib.layers.xavier_initializer())
  conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='VALID')
  biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.1))
  weight=tf.reduce_sum(kernel)/(3*3*48*64)
  biases_ave=tf.reduce_sum(biases)/64
  bias = tf.nn.bias_add(conv, biases)
  conv4 = tf.nn.relu(bias)
  tf.summary.scalar('conv4' + '/weight', weight)
  tf.summary.scalar('conv4' + '/biases', biases_ave)
  tf.summary.histogram('conv4' + '/activations', conv4)
  
  # conv5
  with tf.variable_scope('conv5') as scope:
  conv5=tf.placeholder("float")
  conv=tf.placeholder("float")
  bias=tf.placeholder("float")
  kernel = tf.get_variable('weights',
  shape=[3, 3, 64, 128],
  initializer=tf.contrib.layers.xavier_initializer())
  conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='VALID')
  biases = tf.get_variable('biases', shape=[128], initializer=tf.constant_initializer(0.1))
  weight=tf.reduce_sum(kernel)/(3*3*64*64)
  biases_ave=tf.reduce_sum(biases)/128
  bias = tf.nn.bias_add(conv, biases)
  conv5 = tf.nn.relu(bias)
  tf.summary.scalar('conv5' + '/weight', weight)
  tf.summary.scalar('conv5' + '/biases', biases_ave)
  tf.summary.histogram('conv5' + '/activations', conv5)
  
  # local3
  with tf.variable_scope('local3') as scope:
  
  local3=tf.placeholder("float")
  dim=tf.placeholder(tf.int32)
  bias=tf.placeholder("float")
  weights=tf.placeholder("float")
  reshape = tf.reshape(conv5, [64,-1]) #-1代表缺省
  dim = reshape.get_shape()[1].value
  weights = tf.get_variable('weights', shape=[dim, 500],
  initializer=tf.contrib.layers.xavier_initializer())
  biases = tf.get_variable('biases', shape=[500],
   initializer=tf.constant_initializer(0.1))
  bias = tf.matmul(reshape, weights)+biases
  local3=tf.nn.dropout(local3,0.5)
  local3=tf.nn.relu(bias)
  tf.summary.scalar('local3' + '/weight', tf.reduce_sum(weights)/(dim*100))
  tf.summary.scalar('local3' + '/biases', tf.reduce_sum(biases)/100)
  tf.summary.histogram('local3' + '/activations', local3)
  

  # local4
  with tf.variable_scope('local4') as scope:
  local4=tf.placeholder("float")
  weights=tf.placeholder("float")
  weights = tf.get_variable('weights', shape=[500, 100],
  initializer=tf.contrib.layers.xavier_initializer())
  biases = tf.get_variable('biases', shape=[100], initializer=tf.constant_initializer(0.1))
  local4 = tf.nn.relu(tf.matmul(local3, weights) + biases)
  tf.summary.scalar('local4' + '/weight', tf.reduce_sum(weights)/(500*100))
  tf.summary.scalar('local4' + '/biases', tf.reduce_sum(biases)/100)
  tf.summary.histogram('local4' + '/activations', local4)
 
  #local5
  with tf.variable_scope('local5') as scope:
  local5=tf.placeholder("float")
  weights=tf.placeholder("float")
  weights = tf.get_variable('weights', shape=[100, 20],
  initializer=tf.contrib.layers.xavier_initializer())
  biases = tf.get_variable('biases', shape=[20], initializer=tf.constant_initializer(0.1))
  local5 = tf.nn.relu(tf.matmul(local4, weights) + biases)
  tf.summary.scalar('local5' + '/weight', tf.reduce_sum(weights)/(20*100))
  tf.summary.scalar('local5' + '/biases', tf.reduce_sum(biases)/20)
  tf.summary.histogram('local5' + '/activations', local5)
  

  with tf.variable_scope('local6') as scope:
  local6=tf.placeholder("float")
  weights=tf.placeholder("float")
  weights = tf.get_variable('weights', shape=[20, 4],
  initializer=tf.contrib.layers.xavier_initializer())
  biases = tf.get_variable('biases', shape=[4], 
    initializer=tf.constant_initializer(0.1))
  local6 = tf.matmul(local5, weights) + biases
  
  tf.summary.scalar('local6' + '/weight', tf.reduce_sum(weights)/(20))
  tf.summary.scalar('local6' + '/biases', tf.reduce_sum(biases))
  
  #local6=local6[...,0]
  return local6
def compute_loss(logits, labels):
  loss=tf.placeholder("float")
  loss = tf.reduce_sum(tf.pow(tf.subtract(labels,logits), 2))/128
  tf.summary.histogram('labels' + '/activations', labels)
  tf.summary.histogram('local6' + '/activations', logits)
  tf.summary.scalar('loss', loss)
  tf.summary.histogram('local6-labels' + '/activations', 
    tf.subtract(logits,labels))
  return loss

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
  global_step = tf.Variable(0, trainable=False)

  # Get images and labels for CIFAR-10.
  images=tf.placeholder("float",shape=[None,256,320,3])
  labels=tf.placeholder("float",shape=[None])
  local6=tf.placeholder("float",shape=[None])
  images, labels = inputs()
  logits = inference(images)
  loss = compute_loss(logits, labels)
  lr=0.001
  tf.summary.scalar('learning_rate', lr)
  opt = tf.train.GradientDescentOptimizer(lr)
  grads = opt.compute_gradients(loss)
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  train_op=apply_gradient_op

  # Create a saver.
  saver = tf.train.Saver(tf.global_variables())

  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.summary.merge_all()

  # Build an initialization operation to run below.
  init = tf.global_variables_initializer()

  # Start running operations on the Graph.
  sess = tf.Session(config=tf.ConfigProto(
  log_device_placement=False))
  sess.run(init)

  # Start the queue runners.
  tf.train.start_queue_runners(sess=sess)

  summary_writer = tf.summary.FileWriter('/home/fzyue/Desktop/caffeendtoend/1', sess.graph)

  for step in xrange(500000):
  start_time = time.time()
  _, loss_value = sess.run([train_op, loss])
  duration = time.time() - start_time

  assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

  if step % 10 == 0:
  num_examples_per_step = 64
  examples_per_sec = num_examples_per_step / duration
  sec_per_batch = float(duration)

  format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
  'sec/batch)')
  print (format_str % (datetime.now(), step, loss_value,
  examples_per_sec, sec_per_batch))
  #print(labels)
  #print (sess.run(logits))

  if step % 100 == 0:
  summary_str = sess.run(summary_op)
  summary_writer.add_summary(summary_str, step)

  # Save the model checkpoint periodically.
  if step % 1000 == 0 or (step + 1) == 500000:
  checkpoint_path = os.path.join('/home/fzyue/Desktop/caffeendtoend/1', 'model.ckpt')
  saver.save(sess, checkpoint_path, global_step=step)

def main():
  if tf.gfile.Exists('/home/fzyue/Desktop/caffeendtoend/1'):
  tf.gfile.DeleteRecursively('/home/fzyue/Desktop/caffeendtoend/1')
  tf.gfile.MakeDirs('/home/fzyue/Desktop/caffeendtoend/1')
  train()
main()