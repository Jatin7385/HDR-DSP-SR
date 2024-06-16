import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Get the data to test on : 
# data_path = "C:\\Users\\Jatin\\Desktop\\Engineering\\ISRO Internship\\WORK\\NEW_ARCHITECTURE\\HDR-DSP-SR-main\\Dataset\\test\\15\\"

# imgs = np.load(data_path + "testLR.npy").astype(np.float32)
# ratios = np.load(data_path + "testRatio.npy").astype(np.float32)




path = "./TrainHistory/FNet_Pretraining_Testing/loss_checkpoints.csv"
# files = os.listdir(path)

# for file in files:


df = pd.read_csv(path)

loss = []
epochs = []
for i in range(len(df)):
    print(df.loc[i])
    loss.append(df.loc[i]['TrainLoss'])
    epochs.append(df.loc[i]['Epoch'])

plt.plot(epochs, loss)
plt.show()


