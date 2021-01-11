The round2 prediction in the main directory is computed with the LSTM method. This is our best performing approach which generated our highest ranked submission.

**ARIMA:**
Required packages: 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

To get prediction results, simply follow the respective notebook

**LSTM:**
Required Packages:

import numpy as np
import pandas as pd
import matplotlib
import warnings
import tensorflow as tf
import datetime
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras import optimizers 
from keras.layers import Input
from keras import layers
from keras.models import Model


**Notes for getting the correct prediction results for round 1 and round 2:**

Current version of py script is for round2

For round 1, line 132 in seq2seq attention.py needs to be 28/n_out. For round 2, it is 14/n_out.

Also for round 1, lines 76 -79 are the following (slight changes from round2):
attention_d = layers.dot([decoder_c, h1_c], axes = [2,2])
attention_d = layers.Activation('softmax')(attention_d)
context_d = layers.dot([attention_d, h1_d], axes = [2,1])
decoder_and_context_d = layers.Concatenate(axis=2)([context_d, decoder_d])

To produce the best result we have on kaggle, you need to run the round 1 notebook with the above settings and with n_input as 14,21,28 then take average of the results from these three.


To get prediction results for Round 2, please follow the 145_speedup_final_pred notebook. There are two 
To get prediction results for Round 1, please follow the covid_seq2seq_round1 notebook. To generate prediction not using ensemble, follow the notebook up until the *Prepare output predicitions csv* block. To generate ensemble prediction, run the notebook until the *Prepare output predicitions csv* block for three times using the aforementioned parameter(you may need to change the output names to produce three separate files) then run the ensemble step.


FBProphet:
Run the file:

The covid.ipynb is a Jupiter notebook. To run everything in the notebook, please first 

Pip install fbprophet

Run "import data" section

Then, to get the prediction from the fbprophet model, please run and file and skip the section called "average ( used in round 1; used after this model generated a result)"

To train the model and get predictive result, only run "train, used in round 1 and round 2, basically the same.Only changed the length of the array"

Please notice that part of the codes that are unique to round 1 in commented out. Please use them instead when you need to run and generate the result for round 1.

All the following tuning part will not be used, as they do not generate better results.

After having the result from the fbprophet model and the result from lsma model, please use the "average" section that we just skipped to generate the final result.

