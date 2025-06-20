import numpy as np, matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 1. read csv and get I、Q
x = np.loadtxt('datasets/DPA_200MHz/train_input.csv',  delimiter=',', skiprows=1)
y = np.loadtxt('datasets/DPA_200MHz/train_output.csv', delimiter=',', skiprows=1)

# 2. calculate amplitude（欧几里得范数）——AM/AM
amp_in  = np.hypot(x[:,0], x[:,1])      # = √(I_in² + Q_in²)
amp_out = np.hypot(y[:,0], y[:,1])      # = √(I_out² + Q_out²)

# plt.subplot(1,2,1);
plt.scatter(amp_in, amp_out, s=8); plt.xlabel('|x|'); plt.ylabel('|y|'); plt.title('AM/AM')
plt.tight_layout(); plt.show()