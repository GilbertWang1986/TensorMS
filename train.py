from __future__ import print_function
from __future__ import division


import os
import json
import random
import numpy as np
from MsNN import MsNN


if __name__ == '__main__':


    targets = [line.strip() for line in open('CAS.txt').readlines()]

    for target in targets:

        nn = MsNN( name = target, 
                   target = target,
                   use_PCA = True, 
                   n_components = 256,
                   pca_model_path = 'pca_%s_256.model'%(target),
                   train_pca = True,
                   n_hidden_layers = [256, 128, 64] )


        nn.load_data( data_root_dir = 'train_data/', 
                      ratio_of_test = 0.2 )


        
        nn.train( loss_min=0, 
                  learning_rate=0.01, 
                  keep=0.6, 
                  step_max=5000, 
                  model_path='./nn_%s'%(target) )
        accuracy = nn.validate()
        print(5000, target, accuracy)

        
        # nn.load_pca_model('pca_%s_256.model'%(target))
        # nn.load_model('./nn_%s'%(target))


        i = 0
        step_max = 100
        while True:
            nn.train( loss_min=0, 
                      learning_rate=0.001, 
                      keep=0.6, 
                      step_max=step_max, 
                      model_path='./nn_%s'%(target) )
            i = i+1
            accuracy = nn.validate()
            print(i*step_max, target, accuracy)
            if accuracy >= 0.85:
                print('Stop training at %d, accuracy=%f'%(i*step_max, accuracy))
                break
            if accuracy == 0.0:
                break

