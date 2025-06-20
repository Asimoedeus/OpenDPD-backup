import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# read csv
data = pd.read_csv('datasets/DPA_200MHz/train_input.csv')

#head of table
print(data.columns)  #  ['I', 'Q']


iq_samples = data['I'] + 1j * data['Q']

plt.scatter(np.real(iq_samples), np.imag(iq_samples), s=1)
plt.title("QAM Constellation")
plt.show()
