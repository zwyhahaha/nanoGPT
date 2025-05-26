import numpy as np
import pandas as pd
import argparse
import csv
import os
import glob
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

colorblindbright = [(252,145,100),(188,56,119),(114,27,166)]
colorblinddim    = [(213,167,103),(163,85,114),(104,59,130)]
# colorblindextra  = [(100,200,252),(119,188,56),(166,114,27)]
colorblindextra  = [(0, 128, 128),(123, 104, 238),(189, 183, 107)]

for i in range(len(colorblindbright)):
    r, g, b = colorblindbright[i]
    colorblindbright[i] = (r / 255., g / 255., b / 255.)
for i in range(len(colorblinddim)):
    r, g, b = colorblinddim[i]
    colorblinddim[i] = (r / 255., g / 255., b / 255.)
for i in range(len(colorblindextra)):
    r, g, b = colorblindextra[i]
    colorblindextra[i] = (r / 255., g / 255., b / 255.)

colors = {'sgd':colorblinddim[0], 'sgdn':colorblinddim[1],'adam':colorblinddim[2], 'adamw':colorblinddim[2], \
          'sgd_hd':colorblindbright[0], 'sgdn_hd':colorblindbright[1],'adam_hd':colorblindbright[2],\
          'sgd_hdn':colorblindextra[0],'sgdn_hdn':colorblindextra[1],'adam_hdn':colorblindextra[2]}
names = {'sgd':'SGD','sgdn':'SGDN','adam':'Adam','adamw':'AdamW','sgd_hd':'SGD-HD','sgdn_hd':'SGDN-HD','adam_hd':'Adam-HD'}
linestyles = {'sgd':'--','sgdn':'--','adam':'--','adamw':'--','sgd_hd':':','sgdn_hd':':','adam_hd':':','sgd_hdn':'-','adam_hdn':'-','sgdn_hdn':'-'}
linedashes = {'sgd':[3,3],'sgdn':[3,3],'adam':[3,3],'adamw':[3,3],'sgd_hd':[1,1],'sgdn_hd':[1,1],'adam_hd':[1,1],\
              'sgd_hdn':[10,1e-9],'sgdn_hdn':[10,1e-9],'adam_hdn':[10,1e-9]}

parser = argparse.ArgumentParser(description='Plotting for hypergradient descent PyTorch tests', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--plotDir', help='directory to save the plots', default='plots', type=str)
opt = parser.parse_args()

files = [
    'results/shakespeare-char/+1e-03_+0e+00/adamw.csv',
]

os.makedirs(opt.plotDir, exist_ok=True)

model_titles = {'shakespeare-char': 'babyGPT (on shakespeare-char)'}

data = {}
data_epoch = {}
selected = []
model = None
for fname in files:
    parts = fname.split(os.sep)
    if len(parts) >= 4:
        # results/logreg/+1e-03_+1e-03/sgd_hd.csv
        model_dir = parts[-3]
        param_dir = parts[-2]
        opt_name = os.path.splitext(parts[-1])[0]
        name = f"{opt_name}_{param_dir}"
    else:
        name = os.path.splitext(os.path.basename(fname))[0]
    data[name] = pd.read_csv(fname)
    data_epoch[name] = data[name][pd.notna(data[name].LossEpoch)]
    selected.append(name)
    if model is None:
        parts = fname.split(os.sep)
        for m in model_titles:
            if m in parts:
                model = m
                break
if model is None:
    model = 'model'

plt.figure(figsize=(5,9))
fig = plt.figure(figsize=(5, 9))

ax = fig.add_subplot(211)
for name in selected:
    opt_type = name.split('+')[0][:-1]
    plt.plot(data_epoch[name].Epoch.values, data_epoch[name].LossEpoch.values,label=name,color=colors.get(opt_type, 'k'),linestyle=linestyles.get(opt_type, '-'),dashes=linedashes.get(opt_type, [1,0]))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel('Training loss')
plt.yscale('log')
plt.tick_params(labeltop=False, labelbottom=False, bottom=False, top=False, labelright=False)
plt.grid()
inset_axes(ax, width="50%", height="35%", loc=1)
for name in selected:
    opt_type = name.split('+')[0][:-1]
    plt.plot(data[name].Iteration.values, data[name].Loss.values,label=name,color=colors.get(opt_type, 'k'),linestyle=linestyles.get(opt_type, '-'),dashes=linedashes.get(opt_type, [1,0]))
plt.yticks(np.arange(0, 2.01, 0.5))
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.xscale('log')
plt.xlim([1,9000])
plt.grid()

ax = fig.add_subplot(212)
for name in selected:
    opt_type = name.split('+')[0][:-1]
    plt.plot(data_epoch[name].Epoch.values, data_epoch[name].ValidLossEpoch.values,label=name,color=colors.get(opt_type, 'k'),linestyle=linestyles.get(opt_type, '-'),dashes=linedashes.get(opt_type, [1,0]))
plt.xlabel('Epoch')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel('Validation loss')
plt.yscale('log')
handles, labels = plt.gca().get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
plt.legend(handles,labels,loc='upper right',frameon=1,framealpha=1,edgecolor='black',fancybox=False)
plt.grid()

# plt.tight_layout()
plt.savefig('{}/{}.pdf'.format(opt.plotDir, model), bbox_inches='tight')
