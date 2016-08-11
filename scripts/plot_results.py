import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from adjustText import adjust_text

tags = ['multi','de','it','en','nl']
tags = ['it']
nt = ['0M','4M','40M']
colors = ['#9BBB59','#C0504D','#4F81BD']
dists_y = [0.02, 0.009, 0.02]
dists_x = [10, 10, 10]

legend_dict = {
    'en': 'ML-CNN English',
    'it' : 'SL-CNN Italian',
    'nl' : 'ML-CNN Dutch',
    'de' : 'ML-CNN German',
    'multi' : 'FML-CNN'
}

dir = '../results'
texts = []
for tag in tags:
    path_tag = os.path.join(dir, tag)
    pp = PdfPages('../plots/{}_sl.pdf'.format(tag))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for n, c, d_x,d_y in zip(nt, colors, dists_x,dists_y):
        path = os.path.join(path_tag,'filtered_italian_{}'.format(n))
        path = os.path.join(path,'f1_epoch_history')

        y = np.load(path)
        y_max = max(y)
        y_max_idx = y.argmax()

        ax.plot(y,label='Distant Phase using {} Tweets'.format(n),color=c,lw=1.9)
        ax.plot([y_max_idx], [y_max], 'o', color=c, markersize=7, zorder=2)
        texts.append(ax.text(y_max_idx - d_x, y_max + d_y, "{:10.2f}".format(y_max*100), fontsize=14))

    legend = plt.legend(loc=4, borderaxespad=0.2,frameon=True)
    legend.get_frame().set_facecolor('#FFFFFF')
    legend.get_frame().set_linewidth(0.0)

    plt.title(legend_dict[tag],y=1.03,fontsize=20, fontweight='bold')

    frame1 = plt.gca()
    #frame1.axes.get_yaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #ax.spines['left'].set_visible(False)
    pp.savefig()
    pp.close()
    plt.close()