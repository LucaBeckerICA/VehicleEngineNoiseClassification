import keras
import numpy as np
import keras.backend as K
import src.file_loader as fl
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def accuracy(y_true, y_pred):

    # Compute classification accuracy with a fixed threshold on distances (keras)
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def compute_accuracy(y_true, y_pred):
    # Compute classification accuracy with a fixed threshold on distances (numpy)
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


def create_model():

    '''
    Subnetwork definition (cf. Becker et al.: Fig. 1, Section 4.2)
    '''
    model = keras.Sequential([
        keras.layers.Reshape((32,32,1), input_shape=(32,32)),
        keras.layers.Conv2D(filters=16,kernel_size=[5,5],activation=keras.activations.relu),
        keras.layers.MaxPooling2D(pool_size=[2,2]),
        keras.layers.Flatten(),
        keras.layers.Dense(units=256,activation=keras.activations.relu),
        keras.layers.Dropout(0.8),
        keras.layers.Dense(units=16, activation=keras.activations.softmax)
    ])
    return model


def train_generator(x_train, y_train):


    '''
    Training Generator Routine for the Training Process. Even though, the routine used in Becker et. al.
    can be easily adopted by using the fit() method, a generator simplifies possible changes that might
    improve the system performance.
    '''
    x_train_siam = x_train
    y_train_siam = y_train

    # Counter variable is set to 0 (cf. Training Routine in Becker et. al.)
    counter = 0
    while True:


        # Deprectated Condition: For advanced use only
        if counter == len(x_train):
            shuffle_indices = np.arange(x_train_siam.shape[0])
            np.random.shuffle(shuffle_indices)
            x_train_siam = x_train_siam[shuffle_indices]
            y_train_siam = y_train_siam[shuffle_indices]
            counter = 0

        # Reference Sample
        current_sample = x_train_siam[counter]
        current_label = y_train_siam[counter]

        # Loop iterating through the dataset
        for i in range(len(x_train_siam)):
            if i != counter:

                # Left side is the reference sample
                left = np.reshape(current_sample, (1,) + current_sample.shape)

                # Right side is the training sample
                right = np.reshape(x_train_siam[i], (1,) + x_train_siam[i].shape)

                # Distance Label can be calculated via LabelA XOR LabelB
                y = np.reshape(np.argmax(current_label) ^ np.argmax(y_train_siam[i]), (1,1))
                yield [left, right], y


def train_nn(file, learning_rate, epochs, subsizes):

    '''
    Method that shall be called in start_script_siamese.py
    inputs: The dataset file (.mat, Modulation Features), the dataset split

    outputs: A single confusion matrix (for one Cross Validation) and its respective accuracy metric
    '''

    # Load and arrange dataset for Siamese Training/Inference
    x_train, y_train, x_val, y_val, x_test, y_test = fl.load_data_validation_percentage(file, subsizes)
    x_train_siam, y_train_siam, x_val_siam, y_val_siam, x_test_siam, y_test_siam, ref_label = fl.arrange_for_siamese(x_train, y_train, x_val, y_val, x_test, y_test)


    # Generate Network (For Modulation Features)
    base_network = create_model()

    # Define left and right inputs
    input_a = Input(shape=[32, 32])
    input_b = Input(shape=[32, 32])

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Distance Metric
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    # Overall model (cf. Becker et al. Fig 1)
    model = Model([input_a, input_b], distance)

    # Adding Optimizer and Loss function
    model.compile(loss=contrastive_loss, optimizer=Adam(learning_rate), metrics=[accuracy])

    # Training Procedure
    model.fit_generator(generator=train_generator(x_train, y_train),
                        steps_per_epoch=int(len(x_train) * (len(x_train) - 1)),
                        epochs=epochs,
                        verbose=2,
                        shuffle=False,
                        validation_data=([x_val_siam[:,0], x_val_siam[:,1]], y_val_siam),
                        callbacks=[
                            ModelCheckpoint(filepath="siam_model.hdf5", monitor="val_accuracy", save_best_only=True)]
                        )
    del model

    # Inference
    model = load_model("siam_model.hdf5", custom_objects={'contrastive_loss': contrastive_loss})
    y_pred = model.predict([x_test_siam[:, 0], x_test_siam[:, 1]])
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

