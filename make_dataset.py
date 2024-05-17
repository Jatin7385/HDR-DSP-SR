import numpy as np  
import os


files = os.listdir("./data/hdr-dsp-real-dataset/crop")
train = files[:int(0.7*len(files))]
validation = files[int(0.7*len(files)):int(0.8*len(files))]
test = files[int(0.8*len(files)):]


print("Total : " , len(files))
print("Train(70%) : " , len(train))
print("Validation(10%) : " , len(validation))
print("Test(20%) : " , len(test))

file_path = "./data/hdr-dsp-real-dataset/"
save_file_path = "./Dataset/"

# Train
train_data = {}
train_ratio = {}
for file in train:
    img = np.load(file_path + "crop/" + file)
    ratio = np.load(file_path + "ratios/" + file)

    try:
        img_l = train_data[str(img.shape[0])]
        img_l.append(img)
        train_data[str(img.shape[0])] = img_l


        ratio_l = train_ratio[str(img.shape[0])]
        ratio_l.append(ratio)
        train_ratio[str(img.shape[0])] = ratio_l
    except:
        train_data[str(img.shape[0])] = [img]
        train_ratio[str(img.shape[0])] = [ratio]

    # print(img.shape, ratio.shape)


# Validation
validation_data = {}
validation_ratio = {}
for file in validation:
    img = np.load(file_path + "crop/" + file)
    ratio = np.load(file_path + "ratios/" + file)

    try:
        img_l = validation_data[str(img.shape[0])]
        img_l.append(img)
        validation_data[str(img.shape[0])] = img_l


        ratio_l = validation_ratio[str(img.shape[0])]
        ratio_l.append(ratio)
        validation_ratio[str(img.shape[0])] = ratio_l
    except:
        validation_data[str(img.shape[0])] = [img]
        validation_ratio[str(img.shape[0])] = [ratio]


# Test
test_data = {}
test_ratio = {}
for file in test:
    img = np.load(file_path + "crop/" + file)
    ratio = np.load(file_path + "ratios/" + file)

    try:
        img_l = test_data[str(img.shape[0])]
        img_l.append(img)
        test_data[str(img.shape[0])] = img_l


        ratio_l = test_ratio[str(img.shape[0])]
        ratio_l.append(ratio)
        test_ratio[str(img.shape[0])] = ratio_l
    except:
        test_data[str(img.shape[0])] = [img]
        test_ratio[str(img.shape[0])] = [ratio]




# Training
train_total = 0
for key in train_data.keys():
    file_path_ = save_file_path + f"train/{train_data[key][0].shape[0]}/"
    os.makedirs(file_path_)
    np.save(file_path_ + "trainLR.npy", np.array(train_data[key]))
    np.save(file_path_ + "trainRatio.npy", np.array(train_ratio[key]))
    print("Train : " , np.array(train_data[key]).shape)

    train_total += len(train_data[key])

# Validation
validation_total = 0
for key in validation_data.keys():
    file_path_ = save_file_path + f"val/{validation_data[key][0].shape[0]}/"
    os.makedirs(file_path_)
    np.save(file_path_ + "valLR.npy", np.array(validation_data[key]))
    np.save(file_path_ + "valRatio.npy", np.array(validation_ratio[key]))
    print("Validation : " , np.array(validation_data[key]).shape)

    validation_total += len(validation_data[key])

# Test
test_total = 0
for key in test_data.keys():
    file_path_ = save_file_path + f"test/{test_data[key][0].shape[0]}/"
    os.makedirs(file_path_)
    np.save(file_path_ + "testLR.npy", np.array(test_data[key]))
    np.save(file_path_ + "testRatio.npy", np.array(test_ratio[key]))
    print("Test : " , np.array(test_data[key]).shape)

    test_total += len(test_data[key])


print("Train total : " , train_total)
print("Validation total : " , validation_total)
print("Test total : " , test_total)