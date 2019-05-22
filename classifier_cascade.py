from __future__ import print_function
from __future__ import division


import os
import json
import random
import numpy as np
from MsNN import MsNN
import matplotlib.pylab as plt


def read_jdx(path):
    cas_rn = ''
    spec = []

    f = open(path, 'r')

    ## read CAS CAS REGISTRY NO
    while 1:
        line = f.readline()
        if len(line) == 0:
            break
        if '##CAS REGISTRY NO' in line:
            break
    cas_n = line.split('=')[1].strip()

    ## read mass spectrum
    while 1:
        line = f.readline()
        if len(line) == 0:
            break
        if '##PEAK TABLE=(XY..XY)' in line:
            break
    while 1:
        line = f.readline()
        if len(line) == 0 or '##END=' in line:
            break
        for dot in [i.split(',') for i in line.split()]:
            spec.append([float(xy) for xy in dot])

    f.close()

    return cas_n, spec



if __name__ == '__main__':


    # standard MS
    specs = {}
    for fname in os.listdir('data/10_60'):
        cas_n, spec = read_jdx('data/10_60/%s'%(fname))
        specs[cas_n] = spec



    # targets to detect
    targets = ['107-16-4', '107-13-1', '157-33-5', '107-01-7']

    # classifiers
    classifier_cascade = {}
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

        classifier_cascade[target] = nn



    fnames = os.listdir('validation_data')
    random.shuffle(fnames)
    for fname in fnames:
        data = json.loads(open('validation_data/%s'%(fname)).read())
        x = np.array(data['x'])
        y = np.array(data['y'])

        results = {}
        for target in targets:
            if classifier_cascade[target].predict(y)[0][0] >= 0.8:  # can be adjusted between 1.0~0.5
                results[target] = True
            else:
                results[target] = False

        hits = 0
        for target in targets:
            if results[target]:
                hits += 1
                

        # plot
        if hits == 3:
            plt.subplot(hits+1, 1, 1).plot(x, y, label='Exp.')
            plt.legend()
            plt.xlim([0, 100])
            plt.yticks([])

            idx = 2
            for target in targets:
                if results[target]:
                    _x = [p[0] for p in specs[target]]
                    _y = [p[1] for p in specs[target]]
                    plt.subplot(hits+1, 1, idx).bar(_x, _y, label='Prediction: %s'%(target))
                    idx += 1
                    plt.legend()
                    plt.xlim([0, 100])
                    plt.yticks([])
            
            
            print('Real Composition: ', data['label'])
            plt.show()

            break






    







