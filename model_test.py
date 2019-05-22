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
                   pca_model_path = 'model/pca_%s_256.model'%(target),
                   train_pca = False,
                   n_hidden_layers = [256, 128, 64] )


        nn.load_pca_model('model/pca_%s_256.model'%(target))
        nn.load_model('model/nn_%s'%(target))

        nn.load_data( data_root_dir = 'train_data/', 
                      ratio_of_test = 0.2 )

        accuracy = nn.validate()
        print(target, accuracy)