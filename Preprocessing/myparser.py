import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from end2end_alphaPowerandiAPF import end2end_alphaPowerandiAPF

varargs = {}

varargs['sourcepath'] =  'C:/Users/glgre/Downloads/TD-BRAIN-SAMPLE/derivatives/'
print('Reading data from: '+ 'C:/Users/glgre/Downloads/TD-BRAIN-SAMPLE/derivatives/')
varargs['participantspath'] = 'C:/Users/glgre/Downloads/TD-BRAIN-SAMPLE'
print('Reading data from: '+ 'C:/Users/glgre/Downloads/TD-BRAIN-SAMPLE')

varargs['preprocpath'] = varargs['sourcepath']+'preprocessed'
if not os.path.exists(varargs['preprocpath']):
    os.mkdir(varargs['preprocpath'])
print('Writing preprocessed data to: '+varargs['preprocpath'])
varargs['resultspath'] = varargs['sourcepath']+'results_manuscript'
if not os.path.exists(varargs['resultspath']):
    os.mkdir(varargs['resultspath'])
print('Writing results to: '+varargs['resultspath'])

varargs['condition']=['EO','EC']
varargs['chans']='Pz'

end2end_alphaPowerandiAPF(varargs)