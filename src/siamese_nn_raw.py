import keras
import numpy as np
import keras.backend as K
import src.file_loader2 as fl
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import load_model
from src.SpatialPyramidPooling import SpatialPyramidPooling
import warnings




def accuracy(y_true, y_pred):

    #Compute classification accuracy with a fixed threshold on distances (keras)
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def compute_accuracy(y_true, y_pred):

    # Compute classification accuracy with a fixed threshold on distanced (numpy)
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def euclidean_distance(vects):

    # The utilized distance metric (L2-Norm)
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):

    # Arranging the shapes so that they fit the implementation
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def create_model(b,f,k,p,d,u):

    '''
    Subnetwork definition (cf. Becker et al.: Fig. 1, Section 4.2)
    '''
    model = keras.Sequential([
        keras.layers.Conv2D(filters=f[0],kernel_size=k[0],activation=keras.activations.relu),
        keras.layers.MaxPooling2D(pool_size=p[0]),
        keras.layers.Conv2D(filters=f[1], kernel_size=k[1], activation=keras.activations.relu),
        keras.layers.MaxPooling2D(pool_size=p[1]),
        SpatialPyramidPooling([2,1]),
        keras.layers.Dense(units=u,activation=keras.activations.relu),
        keras.layers.Dropout(d),
        keras.layers.Dense(units=16, activation=keras.activations.softmax)
    ])
    return model


def train_generator(x_train, y_train):

    x_train_siam = x_train
    y_train_siam = y_train
    counter = 0
    while True:

        if counter == len(x_train):
            shuffle_indices = np.arange(x_train_siam.shape[0])
            np.random.shuffle(shuffle_indices)
            x_train_siam = x_train_siam[shuffle_indices]
            y_train_siam = y_train_siam[shuffle_indices]
            counter = 0
        current_sample = x_train_siam[counter]
        current_label = y_train_siam[counter]
        for i in range(len(x_train_siam)):
            if i != counter:
                left = np.reshape(current_sample, (1,) + current_sample.shape + (1,))
                right = np.reshape(x_train_siam[i], (1,) + x_train_siam[i].shape + (1,))
                y = np.reshape(np.argmax(current_label) ^ np.argmax(y_train_siam[i]), (1,1))
                yield [left, right], y



def val_generator(x_val, y_val, x_ref, y_ref):
    x_val_siam = x_val
    y_val_siam = y_val
    counter = 0
    while True:

        if counter == len(x_val_siam):

            counter = 0
        current_sample = x_val_siam[counter]
        current_label = y_val_siam[counter]
        counter += 1
        left = np.reshape(current_sample, (1,) + current_sample.shape + (1,))
        right = np.reshape(x_ref, (1,) + x_ref.shape + (1,))
        y = np.reshape(np.argmax(current_label) ^ np.argmax(y_ref), (1,1))
        yield [left, right], y


def train_nn(file, learning_rate, epochs, subsizes, b, f, k, p, d, u):

    '''
    Method that shall be called in start_script_siamese_raw.py
    inputs: The dataset file (.mat, Non-Modulation Features), the dataset split

    outputs: A single confusion matrix (for one Cross Validation) and its respective accuracy metric
    '''

    # Load and arrange data for Siamese Training/Inference
    x_train, y_train, x_val, y_val, x_test, y_test = fl.load_data_validation_percentage(file, subsizes)
    x_train_siam, y_train_siam, x_val_siam_left, x_val_siam_right, y_val_siam, x_test_siam_left, x_test_siam_right, y_test_siam, ref_sample, ref_label = fl.arrange_for_siamese(x_train, y_train, x_val, y_val, x_test, y_test)


    # Generate Network (For Non-Modulation Features, esp. Spatial Pyramid Pooling)
    base_network = create_model(b,f,k,p,d,u)
    input_a = Input(shape=(None, 32, 1))
    input_b = Input(shape=(None, 32, 1))

    # Define left and right inputs
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Distance Metric
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    # Overall Model (cf. Becker et al. Fig 1)
    model = Model([input_a, input_b], distance)

    # Adding Optimizer and loss function
    model.compile(loss=contrastive_loss, optimizer=Adam(learning_rate), metrics=[accuracy])

    model.fit_generator(generator=train_generator(x_train, y_train),
                        steps_per_epoch=int(len(x_train) * (len(x_train) - 1)),
                        epochs=epochs,
                        verbose=2,
                        shuffle=False,
                        validation_data=val_generator(x_val, y_val, ref_sample, ref_label),
                        validation_steps=len(x_val),
                        callbacks=[
                           ModelCheckpoint(filepath="siam_model_raw.hdf5", monitor="val_accuracy", save_best_only=True)]
                        )

    del model

    # Inference
    model = load_model("siam_model_raw.hdf5", custom_objects={'contrastive_loss': contrastive_loss, 'SpatialPyramidPooling': SpatialPyramidPooling})
    y_pred = np.zeros(len(x_test))
    for i in range(len(y_pred)):
        y_pred[i] = model.predict([
            np.reshape(x_test_siam_left[i], (1,) + x_test_siam_left[i].shape),
            np.reshape(x_test_siam_right[i], (1,) + x_test_siam_right[i].shape)])

    acc = compute_accuracy(y_test_siam, y_pred)
    print("Accuracy:")
    print(acc)

    # Get the detected class from the detected distance: class = distance XOR ref_label
    conf_mat = np.zeros((2, 2))

    # Computation of the Confusion Matrix
    te_pred = np.zeros(y_pred.shape[0])
    for i in range(len(y_pred)):
        if y_pred[i] < 0.5:
            te_pred[i] = 1
        if y_pred[i] >= 0.5:
            te_pred[i] = 0
    ref_label = ref_label * np.ones(len(te_pred), dtype=np.int)
    temp = np.zeros(len(ref_label))
    for i in range(len(ref_label)):
        temp[i] = int(te_pred[i]) ^ int(ref_label[i])
    te_pred = temp
    te_labels = np.argmax(y_test, axis=1)
    # confusion matrix
    for i in range(len(te_pred)):
        if (te_pred[i] == te_labels[i]):
            conf_mat[int(te_pred[i])][int(te_pred[i])] += 1
        if (te_pred[i] != te_labels[i]):
            conf_mat[int(te_labels[i])][int(te_pred[i])] += 1
    conf_mat_list = list(conf_mat)
    print('Confusion', conf_mat_list)

    return acc, conf_mat_list