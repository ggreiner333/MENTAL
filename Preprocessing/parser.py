import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from end2end_alphaPowerandiAPF import end2end_alphaPowerandiAPF

varargs = {}

main_dir = 'C:/Users/glgre/Downloads/TD-BRAIN-SAMPLE/'

varargs[      'sourcepath'] =  main_dir + 'derivatives'
print('Reading data from: '+ varargs['sourcepath'])
varargs[     'preprocpath'] =  main_dir + 'preprocessed'
varargs['participantspath'] =  main_dir
print('Reading data from: '+ varargs['participantspath'])

if not os.path.exists(varargs['preprocpath']):
    os.mkdir(varargs['preprocpath'])
print('Writing preprocessed data to: ' + varargs['preprocpath'])

varargs['resultspath'] = main_dir + 'results_manuscript'
if not os.path.exists(varargs['resultspath']):
    os.mkdir(varargs['resultspath'])
print('Writing results to: ' + varargs['resultspath'])

varargs['condition']=['EO','EC']
varargs['chans']='Pz'

end2end_alphaPowerandiAPF(varargs)