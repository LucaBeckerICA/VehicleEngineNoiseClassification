import numpy as np
from scipy.io import loadmat

def load_data_validation_percentage(file, subsizes):

    '''Important: balanced dataset (1/2 class 1, 1/2 class 2), even amount of files'''

    data = loadmat(file)
    # 2 classes 1(0:20), 2(21:40)
    #c1_data_list = data["data"][0][0:80]
    #c2_data_list = data["data"][0][80:160]


    # For all except mfccs lkwpkw
    c1_data_list = data["data"][0][0:int(len(data["data"][0])/2)]
    c2_data_list = data["data"][0][int(len(data["data"][0])/2):len(data["data"][0])]




    #Noise Test: -> works (test accuracy 0.5) :-)
    #c1_data_list = np.random.normal(0,1,(80,32,32))
    #c2_data_list = np.random.normal(0,1,(80,32,32))

    #c1_data_list = c1_data_list.tolist()
    #c2_data_list = c2_data_list.tolist()

    val_amount = int(subsizes[0]*len(c1_data_list))
    test_amount = int(subsizes[1]*len(c1_data_list))
    train_amount = len(c1_data_list) - val_amount - test_amount


    c1_labels_list = np.zeros(shape=len(c1_data_list),dtype=np.int).tolist()
    c2_labels_list = np.ones(shape=len(c1_data_list), dtype=np.int).tolist()

    # shuffle:
    indices = np.arange(c1_data_list.shape[0])
    np.random.shuffle(indices)
    c1_data_list = c1_data_list[indices]
    np.random.shuffle(indices)
    c2_data_list = c2_data_list[indices]

    c1_data_list = c1_data_list.tolist()
    c2_data_list = c2_data_list.tolist()



    # for now
    cut_train = train_amount
    c1_training_data_list = c1_data_list[0:cut_train]
    c1_val_data_list = c1_data_list[cut_train:cut_train+val_amount]
    c1_test_data_list = c1_data_list[cut_train+val_amount:len(c1_data_list)]

    c1_training_labels_list = c1_labels_list[0:cut_train]
    c1_val_labels_list = c1_labels_list[cut_train:cut_train+val_amount]
    c1_test_labels_list = c1_labels_list[cut_train+val_amount:len(c1_labels_list)]

    c2_training_data_list = c2_data_list[0:cut_train]
    c2_val_data_list = c2_data_list[cut_train:cut_train+val_amount]
    c2_test_data_list = c2_data_list[cut_train+val_amount:len(c2_data_list)]

    c2_training_labels_list = c2_labels_list[0:cut_train]
    c2_val_labels_list = c2_labels_list[cut_train:cut_train+val_amount]
    c2_test_labels_list = c2_labels_list[cut_train+val_amount:len(c2_labels_list)]

    c1_training_data_list.extend(c2_training_data_list)
    training_data_list = c1_training_data_list
    c1_training_labels_list.extend(c2_training_labels_list)
    training_labels_list = c1_training_labels_list

    c1_val_data_list.extend(c2_val_data_list)
    val_data_list = c1_val_data_list
    c1_val_labels_list.extend(c2_val_labels_list)
    val_labels_list = c1_val_labels_list

    c1_test_data_list.extend(c2_test_data_list)
    test_data_list = c1_test_data_list
    c1_test_labels_list.extend(c2_test_labels_list)
    test_labels_list = c1_test_labels_list


    training_data_list = list(training_data_list)
    training_labels_list = list(training_labels_list)
    val_data_list = list(val_data_list)
    val_labels_list = list(val_labels_list)
    test_data_list = list(test_data_list)
    test_labels_list = list(test_labels_list)

    # One-hot Encoding
    for i in range(len(training_labels_list)):
        training_labels_list[i] = one_hot(training_labels_list[i], 2)

    for i in range(len(val_labels_list)):
        val_labels_list[i] = one_hot(val_labels_list[i], 2)

    for i in range(len(test_labels_list)):
        test_labels_list[i] = one_hot(test_labels_list[i], 2)


    return training_data_list, training_labels_list, val_data_list, val_labels_list, test_data_list, test_labels_list


def arrange_for_siamese(x_train, y_train, x_val, y_val, x_test, y_test):


    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    ref_sample = x_train[0]
    ref_label = np.argmax(y_train[0])

    shuffle_indices = np.arange(len(x_train))
    np.random.shuffle(shuffle_indices)
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    x_train_left = x_train[0:int(len(x_train)/2)]
    x_train_right = x_train[int(len(x_train)/2):len(x_train)]

    y_train_left = y_train[0:int(len(y_train)/2)]
    y_train_right = y_train[int(len(y_train)/2):len(x_train)]

    # Undo the one hot encoding and compute distance labels (1 -> dissimilar, 0 -> similar)
    y_train_dist = np.argmax(y_train_left, axis=1) ^ np.argmax(y_train_right, axis=1)

    x_train_data = np.array([x_train_left, x_train_right])
    x_train_data = np.swapaxes(x_train_data, 0, 1)

    #ref_sample = x_train_left[0]
    #ref_label = np.argmax(y_train_left[0])


    x_val_data = np.array([])
    y_val_dist = np.array([])

    if x_val.shape[0] > 0:
        y_val_dist = np.argmax(y_val, axis=1) ^ ref_label
        x_val_ref = np.zeros((len(x_val),) + ref_sample.shape + (1,))
        #x_val_ref[:] = np.reshape(ref_sample, ref_sample.shape + (1,))
        #x_val_data = [x_val, x_val_ref]
        for i in range(len(x_val)):
            x_val_ref[i] = np.reshape(ref_sample, ref_sample.shape + (1,))
            x_val[i] = np.reshape(x_val[i], x_val[i].shape + (1,))
        x_val_left = x_val
        x_val_right = x_val_ref
        #x_val_data = np.swapaxes(x_val_data, 0, 1)


    y_test_dist = np.argmax(y_test, axis=1) ^ ref_label
    x_test_ref = np.zeros((len(x_test),) + ref_sample.shape + (1,))
    #x_test_ref[:] = np.reshape(ref_sample, ref_sample.shape +  (1,))
    #x_test_data = [x_test, x_test_ref]
    for i in range(len(x_test)):
        x_test_ref[i] = np.reshape(ref_sample, ref_sample.shape + (1,))
        x_test[i] = np.reshape(x_test[i], x_test[i].shape + (1,))
    x_test_left = x_test
    x_test_right = x_test_ref
    #x_test_data = np.swapaxes(x_test_data, 0, 1)


    return x_train_data, y_train_dist, x_val_left, x_val_right, y_val_dist, x_test_left, x_test_right, y_test_dist, ref_sample, ref_label

def arrange_train_for_siamese(x_train, y_train):

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    shuffle_indices = np.arange(len(x_train))
    np.random.shuffle(shuffle_indices)
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    x_train_left = x_train[0:int(len(x_train)/2)]
    x_train_right = x_train[int(len(x_train)/2):len(x_train)]

    y_train_left = y_train[0:int(len(y_train)/2)]
    y_train_right = y_train[int(len(y_train)/2):len(x_train)]

    # Undo the one hot encoding and compute distance labels (1 -> dissimilar, 0 -> similar)
    y_train_dist = np.argmax(y_train_left, axis=1) ^ np.argmax(y_train_right, axis=1)

    x_train_data = np.array([x_train_left, x_train_right])
    x_train_data = np.swapaxes(x_train_data, 0, 1)

    return x_train_data, y_train_dist

def one_hot_encoding(data, num_classes):

    oh_mat = np.zeros((len(data), num_classes))
    for i in range(len(data)):
        for j in range(num_classes):
            if data[i] == j:
                oh_mat[i,j] = 1
    return oh_mat


def one_hot(n, amount):
    oh = np.zeros(amount)
    oh[n] = 1
    return oh
