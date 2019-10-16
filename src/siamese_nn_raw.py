import keras
import numpy as np
import keras.backend as K
import kraftfahrzeuge_final.file_loader2 as fl
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import load_model
import kraftfahrzeuge_final.spp_layer as spp
from kraftfahrzeuge_final.SpatialPyramidPooling import SpatialPyramidPooling
from scipy.io import savemat
import warnings

class BestModelCheckpoint(ModelCheckpoint):

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, second_monitor='val_loss'):
        super(BestModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)
        self.second_monitor = second_monitor
        #self.filepath = filepath
        #self.monitor = monitor
        #self.verbose = verbose
        #self.save_best_only = save_best_only
        #self.save_weights_only = save_weights_only
        #self.mode = mode
        #self.period = period
        # Only if we use 'val_loss' as second monitor
        self.second_monitor_best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                current_second = logs.get(self.second_monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                elif current_second is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.second_monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if current_second < self.second_monitor_best:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s'
                                      % (epoch + 1, self.monitor, self.best,
                                         current, filepath))
                            self.best = current
                            self.second_monitor_best = current_second
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


def accuracy(y_true, y_pred):
    #Compute classification accuracy with a fixed threshold on distances-

    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def dist_to_marg(y_true, y_pred):

    mean_pred = K.mean(y_pred)
    marg = 0.5
    return K.abs(mean_pred - marg)

def accuracy2(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    # O O F
    LIM = 1
    #KEK = K.map_fn(fn=lambda x: tf.cond(x >= 0.5,lambda: 1,lambda: 0), elems=tf.reshape(y_pred, [-1]), dtype=y_pred.dtype)
    return K.mean(K.equal(y_true, K.cast(K.map_fn(fn=lambda x: tf.cond(x >= LIM/2,lambda: float(LIM),lambda: 0.0), elems=tf.reshape(y_pred, [-1]), dtype=y_pred.dtype), dtype=y_true.dtype)))
    #return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def compute_accuracy2(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = np.zeros(len(y_pred))
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5:
            pred[i] = 1
        if y_pred[i] <= 0.5:
            pred[i] = 0
    return np.mean(pred == y_true)

def mean(y_true, y_pred):

    return K.mean(y_pred)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
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

    model = keras.Sequential([
        #keras.layers.Reshape((None,32,1)),
        keras.layers.Conv2D(filters=f[0],kernel_size=k[0],activation=keras.activations.relu),
        #keras.layers.Dropout(0.5),
        keras.layers.MaxPooling2D(pool_size=p[0]),
        keras.layers.Conv2D(filters=f[1], kernel_size=k[1], activation=keras.activations.relu),
        keras.layers.MaxPooling2D(pool_size=p[1]),
        #keras.layers.Flatten(),
        #keras.layers.InputLayer(input_tensor=spp.spatial_pyramid_pool(dimensions=[2,1],mode="max", implementation="kaiming")),
        SpatialPyramidPooling([2,1]),
        keras.layers.Dense(units=u,activation=keras.activations.relu),
        keras.layers.Dropout(d),
        #keras.layers.Dense(units=2,activation=keras.activations.sigmoid),
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


    # Generate Data
    #input_shape = [32,32]
    x_train, y_train, x_val, y_val, x_test, y_test = fl.load_data_validation_percentage(file, subsizes)
    x_train_siam, y_train_siam, x_val_siam_left, x_val_siam_right, y_val_siam, x_test_siam_left, x_test_siam_right, y_test_siam, ref_sample, ref_label = fl.arrange_for_siamese(x_train, y_train, x_val, y_val, x_test, y_test)


    # Generate Network
    base_network = create_model(b,f,k,p,d,u)
    input_a = Input(shape=(None, 32, 1))
    input_b = Input(shape=(None, 32, 1))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    # Keras Compile -> slower, but model can be saved
    model.compile(loss=contrastive_loss, optimizer=Adam(learning_rate), metrics=[accuracy, accuracy2, mean])

    # Direct Tensorflow Compile -> fasetr, but model cannot be saved
    #model.compile(loss=contrastive_loss, optimizer=tf.train.AdamOptimizer(learning_rate), metrics=[accuracy, mean])


    # Model Structure
    #model.summary()

    # Run the model
    #init_g = tf.global_variables_initializer()
    #init_l = tf.local_variables_initializer()
    #with K.get_session() as sess:
    #    sess.run(init_g)
    #    sess.run(init_l)

    # Better should be a fit_generator that shuffles the training data each epoch

    #hl_model = Model(inputs=model.layers[2].get_layer('reshape_1').input, outputs=model.layers[2].get_layer('dense_2').output)
    #test_hl_features_before = hl_model.predict(np.array(x_test))
    #savemat("test_hl_features_before.mat", {"data": test_hl_features_before})
    model.layers[2].summary()
    model.fit_generator(generator=train_generator(x_train, y_train),
                        steps_per_epoch=int(len(x_train) * (len(x_train) - 1)),
                        epochs=epochs,
                        verbose=2,
                        shuffle=False,
                        validation_data=val_generator(x_val, y_val, ref_sample, ref_label),
                        validation_steps=len(x_val),
                        #validation_data=([x_val_siam_left[:], x_val_siam_right[:]], y_val_siam),
                        callbacks=[
                           # BestModelCheckpoint(filepath="siam_model.hdf5",monitor='val_accuracy2',save_best_only=True,second_monitor='val_dist_to_marg'),
                           ModelCheckpoint(filepath="siam_model_raw.hdf5", monitor="val_accuracy", save_best_only=True)
                        ]
                        )

    #hl_model = Model(inputs=model.layers[2].get_layer('reshape_1').input, outputs=model.layers[2].get_layer('dense_2').output)
    #test_hl_features_after = hl_model.predict(np.array(x_test))
    #savemat("test_hl_features_after.mat", {"data": test_hl_features_after})
    del model
    model = load_model("siam_model_raw.hdf5", custom_objects={'contrastive_loss': contrastive_loss, 'mean': mean, 'accuracy2': accuracy2, 'SpatialPyramidPooling': SpatialPyramidPooling})
    # Testing the model
    y_pred = np.zeros(len(x_test))
    for i in range(len(y_pred)):
        y_pred[i] = model.predict([
            np.reshape(x_test_siam_left[i], (1,) + x_test_siam_left[i].shape),
            np.reshape(x_test_siam_right[i], (1,) + x_test_siam_right[i].shape)])

    #y_pred = model.predict([x_test_siam_left[:], x_test_siam_right[:]])
    print(y_pred)
    print(y_test_siam)
    acc = compute_accuracy2(y_test_siam, y_pred)
    print("Accuracy:")
    print(acc)

    # Get the detected class from the detected distance: class = distance XOR ref_label
    conf_mat = np.zeros((2, 2))

    # round y_pred
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


'''
epochs = 1
learning = 0.0001
b = 1
f = [16, 1]
k = [[5,5], [3,3]]
p = [[2,2],[2,2]]
d = 0.6
u = 128


acc, conf_mat_list = train_nn('modpcen_dieselbenzin_N256fs16kHz_large_balanced.mat',learning,epochs,[0.0,0.2],b,f,k,p,d,u)
'''