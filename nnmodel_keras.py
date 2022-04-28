import tensorflow as tf
import numpy as np
import pandas as pd
#from keras import backend as K
from utils.tensorflow_utils import get_augmentation, single_class_split, tf_accuracy, tf_precision, tf_recall, tf_f1
from utils.loader import Loader
from utils.trainer import Trainer
from tensorboard.backend.event_processing import event_accumulator
import tensorflow.keras as tk  # pylint: disable=E0401
import tensorflow.keras.layers as tkl  # pylint: disable=E0401
import tensorflow.keras.backend as tkb  # pylint: disable=E0401
import keras_tuner as kt


# Define placeholders and identities
# training = tf.compat.v1.placeholder(False, (), name='training')
# apply_dropout = tf.compat.v1.placeholder(training, (), name='apply_dropout')


def get_label(label_name, dataset):
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    return label


# Define ReLu
def act(x): return tf.nn.leaky_relu(x, 0.1)


# Define regularization
def reg(l1=0.0, l2=0.3): return tf.keras.regularizers.L1L2(l1, l2)



# Function to train model
def train_model(model, dataset_input, epochs, dataset_labels, validation_data=None,
                batch_size=None, callbacks=None):
    # Train the model by feeding it data.

    # Split the dataset into features and label.
    #features = {name: np.array(value) for name, value in dataset.items()}
    #label = np.array(features.pop(label_name))
    # VALIDATION DATA IS MEANT TO BE A TUPLE (X,Y)		
    history = model.fit(x=dataset_input, y=dataset_labels, batch_size=batch_size,
                        callbacks=callbacks, validation_data=validation_data, epochs=epochs, shuffle=False)

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch
    #print(history.history.keys())
    name = list(history.history.keys()) # 5th key is sparse cathegorical accuracy
    #out = hist[name[4]]	# Sparse Cathegorical Accuracy
    #return out[49]
    #hist = pd.DataFrame(history.history)
    loss = history.history[name[0]]
    rbce = history.history[name[3]]
    vloss = history.history[name[7]]
    vbce = history.history[name[10]]
    acc = history.history[name[13]]

    return epochs, loss, rbce, vloss, vbce, acc
# Training function for old Bayesian Optimisation implementation
def train_model_bay_opt(model, dataset_input, epochs, dataset_labels, validation_data=None,
                batch_size=None, callbacks=None):
    # Train the model by feeding it data.

    # Split the dataset into features and label.
    #features = {name: np.array(value) for name, value in dataset.items()}
    #label = np.array(features.pop(label_name))
    # VALIDATION DATA IS MEANT TO BE A TUPLE (X,Y)		
    history = model.fit(x=dataset_input, y=dataset_labels, batch_size=batch_size,
                        callbacks=callbacks, validation_data=validation_data, epochs=epochs, shuffle=False)

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch
    #print(history.history.keys())
    
    #hist = pd.DataFrame(history.history)
    #loss = hist['loss']
    #rbce = hist['reshape_binary_crossentropy']
    #vloss = hist['val_loss']
    #vbce = hist['val_reshape_binary_crossentropy']
    #acc = history.history['val_reshape_sparse_categorical_accuracy']

    return history.history



# Define CNN building blocks 
def dense_block_gen(l2_reg=0.01, dropout_rate=0.4):
    def dense_block(x, size, dropout_rate=dropout_rate):
        x = tkl.Dense(size, kernel_regularizer=tk.regularizers.l2(l2_reg))(x)
        x = tkl.LeakyReLU()(x)
        

        x = tkl.BatchNormalization()(x)
        return tkl.Dropout(rate=dropout_rate)(x)
    return dense_block


def conv_block_gen(l2_reg=0.01, dropout_rate=0.4, monte_carlo=None, bias_initializer=None):
    def conv_block(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), l2_reg=l2_reg):
        x = tkl.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            bias_initializer=bias_initializer,
            kernel_regularizer=tk.regularizers.l2(l2_reg),
        )(x)
        x = tkl.LeakyReLU()(x)
        x = tkl.BatchNormalization()(x)
        return tkl.Dropout(rate=dropout_rate)(x, training=monte_carlo)
    return conv_block

# Class defining Placing and Grasping model
class Placing:
    def __init__(self):
        self.image_shape = {
            'ed': (None, None, 1),
            'rd': (None, None, 1),
            'rc': (None, None, 3),
        }
        self.z_size = 48


         
    # Grasp model used in thesis     
    def define_grasp_model(self, number_primitives: int):
        inputs = [
            tk.Input(shape=self.image_shape['rd'], name='image')
        ]

        conv_block = conv_block_gen(l2_reg=0.001, dropout_rate=0.15) #old dropout_rate 0.35
        conv_block_r = conv_block_gen(l2_reg=0.001, dropout_rate=0.55) #old dropout_rate 0.5

        x = conv_block(inputs[0], 32)
        x = conv_block(x, 32, strides=(2, 2))
        x = conv_block(x, 32)

        x_r = conv_block_r(x, 48)
        x_r = conv_block_r(x_r, 48)

        x_r = conv_block_r(x_r, 64)
        x_r = conv_block_r(x_r, 64)

        x_r = conv_block_r(x_r, 64)
        x_r = conv_block_r(x_r, 48, kernel_size=(2, 2))

        x = conv_block(x, 64)
        x = conv_block(x, 64)

        x = conv_block(x, 108) #old 96
        x = conv_block(x, 108) #old 96

        x = conv_block(x, 128)
        x = conv_block(x, 128, kernel_size=(2, 2))

        reward = tkl.Conv2D(number_primitives, kernel_size=(1, 1), activation='sigmoid', name='reward_grasp')(x_r)
        reward_training = tkl.Reshape((number_primitives,))(reward)

        z_trainings = []
        for i in range(1):
            z = tkl.Conv2D(self.z_size, kernel_size=(1, 1), activity_regularizer=tk.regularizers.l2(0.0005), name=f'z_m{i}')(x)
            z_training = tkl.Reshape((self.z_size,))(z)
            z_trainings.append(z_training)

        outputs = [reward_training] + z_trainings
        return tk.Model(inputs=inputs, outputs=outputs, name='grasp')
    # Unused Place model
    def define_place_model(self):
        inputs = [
            tk.Input(shape=self.image_shape['ed'], name='image_before'),
            tk.Input(shape=self.image_shape['ed'], name='image_goal'),
        ]

        conv_block = conv_block_gen(l2_reg=0.001, dropout_rate=0.35)
        conv_block_r = conv_block_gen(l2_reg=0.001, dropout_rate=0.5)

        x = tkl.Concatenate()(inputs)

        x = conv_block(x, 32)
        x = conv_block(x, 32)

        x = conv_block(x, 32)
        x = conv_block(x, 32)
        x = conv_block(x, 32)
        x = conv_block(x, 32)

        x_r = conv_block_r(x, 32)
        x_r = conv_block_r(x_r, 32)

        x_r = conv_block_r(x_r, 48)
        x_r = conv_block_r(x_r, 48)
        x_r = conv_block_r(x_r, 48)
        x_r = conv_block_r(x_r, 48)

        x_r = conv_block_r(x_r, 48)
        x_r = conv_block_r(x_r, 48)

        x_r = conv_block_r(x_r, 64)
        x_r = conv_block_r(x_r, 48, kernel_size=(2, 2))

        x = conv_block(x, 48)
        x = conv_block(x, 48)

        x = conv_block(x, 64)
        x = conv_block(x, 64)
        x = conv_block(x, 64)
        x = conv_block(x, 64)

        x = conv_block(x, 96)
        x = conv_block(x, 96)

        x = conv_block(x, 128)
        x = conv_block(x, 128, kernel_size=(2, 2))

        reward = tkl.Conv2D(1, kernel_size=(1, 1), activation='sigmoid', name='reward_place')(x_r)
        reward_training = tkl.Reshape((1,))(reward)

        z = tkl.Conv2D(self.z_size, kernel_size=(1, 1), activity_regularizer=tk.regularizers.l2(0.0005), name='z_p')(x)
        z_training = tkl.Reshape((self.z_size,))(z)

        outputs = [reward_training, z_training]
        return tk.Model(inputs=inputs, outputs=outputs, name='place')
    # Unused combination model
    def define_merge_model(self):
        input_shape = (self.z_size)

        z_m = tk.Input(shape=input_shape, name='z_m')
        z_p = tk.Input(shape=input_shape, name='z_p')

        dense_block = dense_block_gen(l2_reg=0.01, dropout_rate=0.2)
        x = z_m - z_p

        x = dense_block(x, 128)
        x = dense_block(x, 128)
        x = dense_block(x, 64)

        reward = tkl.Dense(1, activation='sigmoid', name='reward_merge')(x)
        return tk.Model(inputs=[z_m, z_p], outputs=[reward], name='merge')

    def build_model(self,hp):
        # Function to create hypermodel for keras
        # learning_rate=hp.Float('learning_rate', 1e-5, 1e-3, sampling = 'log')
        model = self.define_grasp_model(3)
        #momentum=hp.Float('momentum', 0.0, 1.0)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=hp.Float('learning_rate', 1e-5, 1e-3, sampling = 'log'),momentum = 0.9),
    			loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryCrossentropy(),tf.keras.metrics.SparseCategoricalAccuracy()])
    
        return model
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, batch_size = hp.Int('batch_size',16,48))
    
    @staticmethod
    def binary_decision(string: str, p: float) -> bool:
        return float(int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16) % 2**16) / 2**16 < p

# Callback management    
class haltCallback(tk.callbacks.Callback):
    def stopping_cond(self, epoch, logs={}):
        if(epoch==2):
            print("Stopping the training")
            self.model.stop_training=True
# History extraction    
def extract_history(best_trial):
    loss = []
    val_loss = []
    bce = []
    val_bce = []
    acc = []
    val_acc = []
    logdir = "C:/Users/User/Documents/UNI/MEng Project/PYFILES/TensorBoard/"
    for set_data in ['train','validation']:
        if set_data == 'train':
            ea = event_accumulator.EventAccumulator(logdir + best_trial + '/execution0/')
            ea.Reload()  
            for i in range(len(ea.Scalars('epoch_loss'))):
                loss.append(ea.Scalars('epoch_loss')[i][2])
                bce.append(ea.Scalars('reshape_binary_crossentropy')[i][2])
                acc.append(ea.Scalars('reshape_sparse_categorical_accuracy')[i][2])
        if set_data == 'validation':
            ea = event_accumulator.EventAccumulator(logdir + best_trial + '/execution0/')
            ea.Reload()  
            for i in range(len(ea.Scalars('epoch_loss'))):
                val_loss.append(ea.Scalars('epoch_loss')[i][2])
                val_bce.append(ea.Scalars('reshape_binary_crossentropy')[i][2])
                val_acc.append(ea.Scalars('reshape_sparse_categorical_accuracy')[i][2]) 
    return loss, val_loss, bce, val_bce, acc, val_acc
# Keras optimiser model
class KerasHyperModel(kt.HyperModel):
    
    def build(self, hp):
        model = Placing()
        return model.build_model(hp)
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, batch_size = hp.Int('batch_size',16,48))