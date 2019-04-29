from __future__ import print_function
from __future__ import division


import os
import json
import random
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA



class MsNN():
    def __init__( self, name, target,
                  data_root_dir, ratio_of_test, 
                  use_PCA, n_components, 
                  n_hidden_layers, keep, learning_rate, nproc):
        
        self.name = name
        self.target = target
        self.use_PCA = use_PCA
        self.keep = keep
        

        # divide the data set 
        fnames = os.listdir(data_root_dir)
        random.shuffle(fnames)
        target_list = [self.is_target(data_root_dir+fname) for fname in fnames]
        fname_positive = [fname for idx, fname in enumerate(fnames) 
                          if target_list[idx]]
        fname_negative = [fname for idx, fname in enumerate(fnames) 
                          if not target_list[idx]][:len(fname_positive)]
        data_fname = fname_positive + fname_negative

        num_test_data = int( len(data_fname)*ratio_of_test )
        test_data_fname  = data_fname[:num_test_data]
        train_data_fname = data_fname[num_test_data:]

        
        # read train data
        self.train_spec, self.train_label = \
        self.read_train_data(data_root_dir+train_data_fname[0])

        for fname in train_data_fname[1:]:
            spec, label = self.read_train_data(data_root_dir+fname)
            self.train_spec  =  np.vstack( (self.train_spec, spec) )
            self.train_label =  np.vstack( (self.train_label, label) )

        # read test data
        self.test_spec, self.test_label = \
        self.read_train_data(data_root_dir+test_data_fname[0])

        for fname in test_data_fname[1:]:
            spec, label = self.read_train_data(data_root_dir+fname)
            self.test_spec  =  np.vstack( (self.test_spec, spec) )
            self.test_label =  np.vstack( (self.test_label, label) )


        # PCA
        if self.use_PCA:
            self.pca = PCA(n_components=n_components, copy=True, whiten=False)
            self.pca.fit(self.train_spec)
            self.train_spec = self.pca.transform(self.train_spec)
            self.test_spec  = self.pca.transform(self.test_spec)


        # construct NN
        self.graph = tf.Graph()
        with self.graph.as_default():
            # data entry
            self.xs        = tf.placeholder(tf.float32, [None,  n_components])
            self.ys        = tf.placeholder(tf.float32, [None,             2])
            self.keep_prob = tf.placeholder(tf.float32)

            # hidden layers
            self.hidden_layers = []
            self.hidden_layers.append( tf.nn.dropout(self.add_layer(self.xs, n_components, n_hidden_layers[0],
                                                                    activation_function=tf.nn.relu),
                                                     self.keep_prob) 
                                     )
            for i in range(len(n_hidden_layers)-1):
                self.hidden_layers.append( tf.nn.dropout(self.add_layer(self.hidden_layers[i], n_hidden_layers[i], n_hidden_layers[i+1],
                                                                        activation_function=tf.nn.relu),
                                                         self.keep_prob)
                                         )

            # output layer
            self.prediction = self.add_layer(self.hidden_layers[-1], n_hidden_layers[-1], 2, activation_function=tf.nn.softmax)

            # loss: cross entropy
            self.loss = -tf.reduce_sum( self.ys*tf.log(self.prediction) )

            # training
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            # save and restore trained model
            self.saver = tf.train.Saver(max_to_keep=1000)

            # initiation
            config = tf.ConfigProto( device_count={"CPU": nproc},
                                     inter_op_parallelism_threads = 1,   
                                     intra_op_parallelism_threads = 1,  
                                     log_device_placement=True )  
            self.sess = tf.Session(graph=self.graph, config=config)
            self.sess.run(tf.global_variables_initializer())




    def train(self, loss_min, step_max=None, model_path=None):
           
        counter = 0
        while True:

            self.sess.run(self.train_step, 
                          feed_dict={self.xs:self.train_spec, self.ys:self.train_label, self.keep_prob:self.keep})
            if counter%100 == 0:
                train_loss = self.sess.run(self.loss, 
                             feed_dict={self.xs:self.train_spec, self.ys:self.train_label, self.keep_prob:1.0})
                test_loss  = self.sess.run(self.loss, 
                             feed_dict={self.xs:self.test_spec,  self.ys:self.test_label, self.keep_prob:1.0})
                print( '%-10d   Train Loss=%-12.6f  Test Loss=%-12.6f'
                       %(counter, train_loss, test_loss) )
            
            # stop training
            if train_loss <= loss_min:
                train_stat = 1
                break
            if step_max != None:
                if counter >= step_max:
                    train_stat = 0
                    break
            counter = counter+1

        # save trained model 
        if not model_path is None:
            self.saver.save(self.sess, model_path)
    
        print('Training is stopped')
        return train_stat




    def load(self, model_path):
        self.saver.restore(self.sess, model_path)




    def save(self, model_path):
        self.saver.save(self.sess, model_path)




    def validate(self):
        predictions = self.sess.run( self.prediction, 
                                     feed_dict={self.xs:self.test_spec, 
                                                self.keep_prob:1.0})
        predictions = [i[0]>i[1] for i in predictions]
        labels = [i[0]>i[1] for i in self.test_label]

        right_predictions = [idx for idx in range(len(predictions)) 
                             if predictions[idx] == labels[idx]]

        return len(right_predictions)/len(predictions)
        



    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        weights = tf.Variable(tf.random_normal([in_size, out_size], stddev = 1.0))*1.0e-3
        biases  = tf.Variable(tf.zeros([1, out_size])+0.1)
        wx_plus_b = tf.matmul(inputs, weights)+biases
        if activation_function == None:
            return wx_plus_b
        else:
            return activation_function(wx_plus_b)




    def read_train_data(self, path):
        data = open(path).read()
        data = json.loads(data)

        spec = np.array(data['y'])
        label = data['label']
        if self.target in label:
            label = np.array( [1.0, 0.0] )
        else:
            label = np.array( [0.0, 1.0] )

        return spec, label




    def is_target(self, path):
        data = open(path).read()
        data = json.loads(data)
        if self.target in data['label']:
            return True
        else:
            return False









## To Do ##
# 1. Save & Reload trained model
# 2. Evaluate single MS spectrum
#
#



if __name__ == '__main__':

    nn_107_02_8 = MsNN( name = '107-02-8', 
                        target = '107-02-8',
                        data_root_dir = 'train_data/', 
                        ratio_of_test = 0.2, 
                        use_PCA = True, 
                        n_components = 128, 
                        n_hidden_layers = [64, 64, 64], 
                        keep = 1.0, 
                        learning_rate = 0.01, 
                        nproc = 1)


    # nn_107_02_8.train(loss_min=0, step_max=100, model_path='./MSNN')
    # nn_107_02_8.load(model_path='./MSNN')


    for i in range(10000):
        nn_107_02_8.train(loss_min=0, step_max=100, model_path='./MSNN')
        print( nn_107_02_8.validate() )



