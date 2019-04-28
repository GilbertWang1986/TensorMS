from __future__ import print_function
from __future__ import division


import os
import json
import random
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA



class MsNN():
    def __init__( self, nproc, data_root_dir, ratio_of_test, 
                  use_PCA, n_components, 
                  n_hidden_layers, keep, learning_rate):
        
        self.data_root_dir = data_root_dir
        self.use_PCA = use_PCA
        self.keep = keep
        data_fname = os.listdir(data_root_dir)
        random.shuffle(data_fname)
        self.train_data_fname    = data_fname[:500]
        self.test_data_fname     = data_fname[500:600]
        self.validate_data_fname = data_fname[600:700]

        # data entry
        self.xs        = tf.placeholder(tf.float32, [None,  n_components])
        self.ys        = tf.placeholder(tf.float32, [None,             2])
        self.keep_prob = tf.placeholder(tf.float32)

        # hidden layers
        self.hidden_layers = []
        self.hidden_layers.append(tf.nn.dropout(self.add_layer(self.xs, n_components, n_hidden_layers[0],
                                                			   activation_function=tf.nn.relu),
        			                            self.keep_prob))
        for i in range(len(n_hidden_layers)-1):
            self.hidden_layers.append(tf.nn.dropout(self.add_layer(self.hidden_layers[i], n_hidden_layers[i], n_hidden_layers[i+1],
                                                    activation_function=tf.nn.relu),
                                                    self.keep_prob))

        # output layer
        self.prediction = self.add_layer(self.hidden_layers[-1], n_hidden_layers[-1], 2, activation_function=tf.nn.softmax)

        # loss
        self.loss = -tf.reduce_sum( self.ys*tf.log(self.prediction) )

        # training
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # save and restore trained model
        self.saver = tf.train.Saver(max_to_keep=1000)

        ## initiation
        config = tf.ConfigProto(device_count={"CPU": nproc},
                                inter_op_parallelism_threads = 1,   
                                intra_op_parallelism_threads = 1,  
                                log_device_placement=True)  
        self.sess = tf.Session(config=config)

        # read train data
        self.train_id, self.train_ms_spec = self.read_train_data(self.data_root_dir+self.train_data_fname[0])
        for fname in self.train_data_fname[1:]:
            identification, ms_spec = self.read_train_data(self.data_root_dir+fname)
            ms_spec = (ms_spec-np.min(ms_spec))/(np.max(ms_spec)-np.min(ms_spec))
            self.train_id      =  np.vstack((self.train_id, identification))
            self.train_ms_spec =  np.vstack((self.train_ms_spec, ms_spec))

        # read train data
        self.test_id, self.test_ms_spec = self.read_train_data(self.data_root_dir+self.test_data_fname[0])
        for fname in self.test_data_fname[1:]:
            identification, ms_spec = self.read_train_data(self.data_root_dir+fname)
            ms_spec = (ms_spec-np.min(ms_spec))/(np.max(ms_spec)-np.min(ms_spec)) 
            self.test_id      =  np.vstack((self.test_id, identification))
            self.test_ms_spec =  np.vstack((self.test_ms_spec, ms_spec))

        if self.use_PCA:
            self.pca = PCA(n_components=n_components, copy=True, whiten=False)
            self.pca.fit(self.train_ms_spec)
            self.train_ms_spec = self.pca.transform(self.train_ms_spec)
            self.test_ms_spec  = self.pca.transform(self.test_ms_spec)


    def close(self):
        self.sess.close()


    def train(self, loss_min, step_max=None):
        self.sess.run(tf.global_variables_initializer())   
        counter = 0
        while True:
            self.sess.run(self.train_step, 
                          feed_dict={self.xs:self.train_ms_spec, self.ys:self.train_id, self.keep_prob:self.keep})
            if counter%100 == 0:
                train_loss = self.sess.run(self.loss, 
                             feed_dict={self.xs:self.train_ms_spec, self.ys:self.train_id, self.keep_prob:1.0})
                test_loss  = self.sess.run(self.loss, 
                             feed_dict={self.xs:self.test_ms_spec,  self.ys:self.test_id, self.keep_prob:1.0})
                print '%-10d   Train Loss=%-12.6f  Test Loss=%-12.6f'%(counter, train_loss, test_loss)
            
            ## stop training
            if train_loss <= loss_min:
                train_stat = 1
                break
            if step_max != None:
                if counter >= step_max:
                    train_stat = 0
                    break
            counter = counter+1
        return train_stat


    def validation(self):
        # validation
        counter_validation = 0.0
        counter_right      = 0.0
        for fname in self.validate_data_fname:
            counter_validation = counter_validation + 1
            identification, ms_spec = self.read_train_data(self.data_root_dir+fname)
            ms_spec = (ms_spec-np.min(ms_spec))/(np.max(ms_spec)-np.min(ms_spec))
            ms_spec = ms_spec.reshape(1,ms_spec.shape[0])

            if self.use_PCA:
                ms_spec = self.pca.transform(ms_spec)

            predicted_identification = self.sess.run(self.prediction, 
                                                     feed_dict={self.xs:ms_spec, self.keep_prob:self.keep})
            
            if (predicted_identification[0,0] > predicted_identification[0,1]) and (identification[0] == 1.0):
                counter_right = counter_right+1.0
            elif (predicted_identification[0,0] < predicted_identification[0,1]) and (identification[1] == 1.0):
                counter_right = counter_right+1.0

        return counter_right/counter_validation
        

    # def evaluate(self):


    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        weights = tf.Variable(tf.random_normal([in_size, out_size], stddev = 1.0))*1.0e-3
        biases  = tf.Variable(tf.zeros([1, out_size])+0.1)
        wx_plus_b = tf.matmul(inputs, weights)+biases
        if activation_function == None:
            return wx_plus_b
        else:
            return activation_function(wx_plus_b)


    def read_train_data(self, path):
        f = open(path)
        line = f.readline()
        if 'not contain' in line:
            identification = np.array([0.0, 1.0])
        else:
            identification = np.array([1.0, 0.0])

        ms_spec = []
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            ms_spec.append(float(line.strip()))            
        ms_spec = np.array(ms_spec)
        f.close()

        return identification, ms_spec








## To Do ##
# 1. Save & Reload trained model
# 2. Evaluate single MS spectrum
#
#
