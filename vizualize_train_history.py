import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Get the data to test on : 
# data_path = "C:\\Users\\Jatin\\Desktop\\Engineering\\ISRO Internship\\WORK\\NEW_ARCHITECTURE\\HDR-DSP-SR-main\\Dataset\\test\\15\\"

# imgs = np.load(data_path + "testLR.npy").astype(np.float32)
# ratios = np.load(data_path + "testRatio.npy").astype(np.float32)




# path = "./TrainHistory/FNet_Pretraining_Testing/loss_checkpoints_to_800_with_checkpoints.csv"
path = "./TrainHistory/AWF_SR_Training/loss_checkpoints_110_epochs.csv"
# files = os.listdir(path)

# for file in files:


df = pd.read_csv(path)

loss = []
epochs = []
minEpoch = 0
for i in range(len(df)):
    print(df.loc[i])
    # if(df.loc[i]["ValWarpLoss"] == 0.1530012406874448):
    #     minEpoch = df.loc[i]['Epoch']
    loss.append(df.loc[i]['ValLoss'])
    epochs.append(df.loc[i]['Epoch'])

print("Minimum Loss reached : " , np.min(np.array(loss)))
print("Epoch at which occured : " , minEpoch)
plt.plot(epochs, loss)
plt.xlabel("Epochs")
plt.ylabel("ValLoss")
plt.title("FNet Pretraining Loss vs Epochs")
plt.show()


