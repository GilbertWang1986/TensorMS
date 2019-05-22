from __future__ import print_function
from __future__ import division


import os
import json
import random
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import matplotlib.pylab as plt




class MsNN():
    def __init__( self, name, target,
                  use_PCA, n_components, pca_model_path, train_pca,
                  n_hidden_layers):
        
        self.name = name
        self.target = target
        self.use_PCA = use_PCA
        self.n_components = n_components
        self.pca_model_path = pca_model_path
        self.train_pca = train_pca


        # construct NN
        self.graph = tf.Graph()
        with self.graph.as_default():
            # data entry
            self.xs        = tf.placeholder(tf.float32, [None,  n_components])
            self.ys        = tf.placeholder(tf.float32, [None,             2])
            self.keep_prob = tf.placeholder(tf.float32)
            self.lr_prob   = tf.placeholder(tf.float32)  # learning rate

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
            self.train_step = tf.train.AdamOptimizer(self.lr_prob).minimize(self.loss)

            # save and restore trained model
            self.saver = tf.train.Saver(max_to_keep=1000)

            # initiation
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())




    def load_data(self, data_root_dir, ratio_of_test, balance=True):
        # divide the data set 
        fnames = os.listdir(data_root_dir)
        random.shuffle(fnames)
        
        if balance == True:
            target_list = [self.is_target(data_root_dir+fname) for fname in fnames]
            fname_positive = [fname for idx, fname in enumerate(fnames) 
                              if target_list[idx]]
            fname_negative = [fname for idx, fname in enumerate(fnames) 
                              if not target_list[idx]][:len(fname_positive)]
            data_fname = fname_positive + fname_negative
        else:
            data_fname = fnames

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
            if self.train_pca:
            	self.pca_fit(data=self.train_spec, pca_model_path=self.pca_model_path)
            else:
                self.load_pca_model(self.pca_model_path)

            self.train_spec = self.pca.transform(self.train_spec)
            self.test_spec  = self.pca.transform(self.test_spec)




    def pca_fit(self, data, pca_model_path):

        self.pca = PCA(n_components=self.n_components, copy=True, whiten=False)
        self.pca.fit(data)

        if not pca_model_path == None:
            joblib.dump(self.pca, pca_model_path)




    def load_pca_model(self, pca_model_path):
        self.pca = joblib.load(pca_model_path)





    def train(self, loss_min, learning_rate, keep, step_max=None, model_path=None):
           
        counter = 0
        while True:

            self.sess.run(self.train_step, 
                          feed_dict={ self.xs:        self.train_spec, 
                                      self.ys:        self.train_label,
                                      self.lr_prob:   learning_rate,
                                      self.keep_prob: keep}
                          )
            
            if counter%100 == 0:
                train_loss = self.sess.run(self.loss, 
                                           feed_dict={ self.xs:        self.train_spec, 
                                                       self.ys:        self.train_label, 
                                                       self.keep_prob: 1.0}
                                           )
                test_loss  = self.sess.run(self.loss, 
                                           feed_dict={ self.xs:        self.test_spec,  
                                                       self.ys:        self.test_label, 
                                                       self.keep_prob: 1.0}
                                           )
                
                # print( '%-10d   Train Loss=%-12.6f  Test Loss=%-12.6f'
                #        %(counter, train_loss, test_loss) )
            
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
        if not model_path == None:
            self.saver.save(self.sess, model_path)
    
        # print('Training is stopped')
        return train_stat




    def load_model(self, model_path):
        self.saver.restore(self.sess, model_path)




    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)




    def validate(self):
        predictions = self.sess.run( self.prediction, 
                                     feed_dict={ self.xs:        self.test_spec, 
                                                 self.keep_prob: 1.0 }
                                    )
        predictions = [i[0]>i[1] for i in predictions]
        labels = [i[0]>i[1] for i in self.test_label]

        right_predictions = [idx for idx in range(len(predictions)) 
                             if predictions[idx] == labels[idx]]

        return len(right_predictions)/len(predictions)




    def predict(self, spec):
        spec = spec.reshape(1, -1)
        spec = self.pca.transform(spec)
        predictions = self.sess.run( self.prediction, 
                                     feed_dict={ self.xs:  spec, 
                                                 self.keep_prob: 1.0 }
                                   )
        return predictions 




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
                        use_PCA = True, 
                        n_components = 256,
                        pca_model_path = 'pca_107_02_8_N256.model',
                        train_pca = True,
                        n_hidden_layers = [256, 128, 64])

    nn_107_02_8.load_model(model_path='./MSNN')
    nn_107_02_8.load_pca_model('pca_107_02_8_N256.model')

    nn_107_02_8.load_data( data_root_dir = 'validation_data/', 
                           ratio_of_test = 0.2 )


    i = 0
    step_max = 100
    while True:
        nn_107_02_8.train(loss_min=0, 
        	              learning_rate=0.001, 
        	              keep=0.6, 
        	              step_max=step_max, 
        	              model_path='./MSNN')
        i = i+1
        accuracy = nn_107_02_8.validate()
        print(i*step_max, accuracy)
        if accuracy >= 0.8:
        	print('Stop training at %d, accuracy=%f'%(i*step_max, accuracy))
        	break



    




