from __future__ import print_function
from __future__ import division


import copy
import random
import json

import numpy as np
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
    


def gaussian_smearing(spec_input, width, x_min, x_max, x_delta, peak_shift=False):

    spec = copy.deepcopy(spec_input)

    ## random shift for MS spectrum
    if peak_shift:
        for i in range(len(spec)):
            if spec[i][1] <= 2000:
                if random.uniform(0.0,1.0) <= 0.25:                #shift 1 randomly discard small peaks (Int.<=1000)
                    spec[i][1] = 0.0
            if spec[i][1] <= 1000:
                if random.uniform(0.0,1.0) <= 0.25:                #shift 1 randomly discard small peaks (Int.<=1000)
                    spec[i][1] = 0.0
            spec[i][0] = spec[i][0]+random.uniform(-0.25,  0.25)  #shift 2 peak position
            spec[i][1] = spec[i][1]*random.uniform( 0.75,  1.25)  #shift 3 peak intensity

    x = np.linspace(x_min, x_max, int((x_max-x_min)/x_delta))
    y = np.zeros(len(x))
    width = -1.0/(2.0*width*width)

    for dot in spec:
        y = y + dot[1] * np.exp( np.power(x-dot[0],2)*width )
    return x,y 



def generator_train_data(specs, cas_ns, n_specs):

    indice = random.sample(range(len(specs)), n_specs)

    label = []
    y = np.zeros_like(np.linspace(0, 100, int((100-0)/0.02)))

    for idx in indice:
        x, _y = gaussian_smearing(specs[idx], 0.1, 0, 100, 0.02, True)
        y = y + _y**random.uniform(0.25, 0.75)
        label.append(cas_ns[idx])

    y = y/np.max(y)

    return x, y, label





if __name__ == '__main__':
    import matplotlib.pylab as plt
    import os

    specs = []
    cas_ns = []
    for fname in os.listdir('data/10_60'):
        cas_n, spec = read_jdx('data/10_60/%s'%(fname))
        specs.append(spec)
        cas_ns.append(cas_n)


    n_specs = 1
    N = 200
    for i in range(N):
        x, y, label = generator_train_data(specs, cas_ns, n_specs)
        data = {}
        data['label'] = label
        data['x'] = x.tolist()
        data['y'] = y.tolist()
        with open('train_data/%s_%s.json'%(n_specs, i), 'w') as f:
            f.write( json.dumps(data) )


    n_specs = 2
    N = 300
    for i in range(N):
        x, y, label = generator_train_data(specs, cas_ns, n_specs)
        data = {}
        data['label'] = label
        data['x'] = x.tolist()
        data['y'] = y.tolist()
        with open('train_data/%s_%s.json'%(n_specs, i), 'w') as f:
            f.write( json.dumps(data) )


    n_specs = 3
    N = 300
    for i in range(N):
        x, y, label = generator_train_data(specs, cas_ns, n_specs)
        data = {}
        data['label'] = label
        data['x'] = x.tolist()
        data['y'] = y.tolist()
        with open('train_data/%s_%s.json'%(n_specs, i), 'w') as f:
            f.write( json.dumps(data) )



    n_specs = 4
    N = 100
    for i in range(N):
        x, y, label = generator_train_data(specs, cas_ns, n_specs)
        data = {}
        data['label'] = label
        data['x'] = x.tolist()
        data['y'] = y.tolist()
        with open('train_data/%s_%s.json'%(n_specs, i), 'w') as f:
            f.write( json.dumps(data) )




    # x, y, indice = generator_train_data(specs, cas_ns, n_specs=4)
    # plt.subplot(2,1,1).plot(x, y)
    # plt.xlim([0, 100])

    # for idx in indice:
    #     plt.subplot(2,1,2).bar([i[0] for i in specs[idx]], [i[1] for i in specs[idx]])
    # plt.xlim([0, 100])

    # plt.show()




    # plot

    # fname = os.listdir('data/10_60')[10]

    # cas_n, spec = read_jdx('data/10_60/%s'%(fname))

    # plt.subplot(2,1,1).bar([i[0] for i in spec], [i[1] for i in spec])
    # plt.xlim([0, 100])
    # x, y = gaussian_smearing(spec, 0.1, 0, 100, 0.01, True)
    # plt.subplot(2,1,2).plot(x, y)
    # plt.xlim([0, 100])
    # plt.show()