
import numpy as np
import pandas as pd
#from keras import backend as K
from utils.tensorflow_utils import get_augmentation, single_class_split, tf_accuracy, tf_precision, tf_recall, tf_f1
from utils.loader import Loader
from utils.trainer import Trainer
import tensorflow.keras as tk  # pylint: disable=E0401
import tensorflow.keras.layers as tkl  # pylint: disable=E0401
import tensorflow.keras.backend as tkb  # pylint: disable=E0401


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

# Old model implementation
def create_model(my_learning_rate, label_name, dataset):
    # Create and compile a simple linear regression model.
    # Most simple tf.keras models are sequential.

    #label = tf.identity(get_label(label_name, dataset), name='label')
    model = tf.keras.models.Sequential()
    #handle = tf.compat.v1.placeholder_with_default(tf.string, shape=(), name='handle')
    #training = tf.compat.v1.placeholder_with_default(False, (), name='training')
    #apply_dropout = tf.compat.v1.placeholder_with_default(training, (), name='apply_dropout')


    # Add the layer containing the feature columns to the model.
    #model.add(my_feature_layer)

    # Describe the topography of the model by calling the tf.keras.layers.Dense
    # method once for each layer. We've specified the following arguments:
    #   * units specifies the number of nodes in this layer.
    #   * activation specifies the activation function (Rectified Linear Unit).
    #   * name is just a string that can be useful when debugging.

    # Define the first hidden layer with 32 nodes.
    model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), activation=act, kernel_regularizer=reg(),
                                     bias_regularizer=reg()))
    # Define regularization
    model.add(tf.keras.layers.BatchNormalization())
    
    # Define dropout
    #model.add(tf.keras.layers.Dropout(0.4))

    # Define the second hidden layer with 48 nodes.
    model.add(tf.keras.layers.Conv2D(48, (5, 5), strides=(1, 1), dilation_rate=(1, 1), activation=act,
                                     kernel_regularizer=reg(), bias_regularizer=reg()))
    # Define regularization
    model.add(tf.keras.layers.BatchNormalization())

    # Define the third hidden layer with 64 nodes.
    model.add(tf.keras.layers.Conv2D(64, (5, 5), dilation_rate=(1, 1), activation=act,
                                     kernel_regularizer=reg(), bias_regularizer=reg()))

    # Define regularization and dropout
    model.add(tf.keras.layers.BatchNormalization())

    # Define dropout
    model.add(tf.keras.layers.Dropout(0.4))

    # Define the fourth hidden layer with 142 nodes.
    model.add(tf.keras.layers.Conv2D(128, (6, 6), dilation_rate=(1, 1), activation=act, kernel_regularizer=reg(),
                                     bias_regularizer=reg()))

    # Define dropout
    model.add(tf.keras.layers.Dropout(0.5))
    

    # Define the fifth hidden layer with 128 nodes.
    model.add(tf.keras.layers.Conv2D(128, (1, 1), dilation_rate=(1, 1), activation=act, kernel_regularizer=reg(),
                                     bias_regularizer=reg()))

    # Define dropout
    model.add(tf.keras.layers.Dropout(0.6))

    model.add(tf.keras.layers.Conv2D(3, (1, 1), dilation_rate=(1, 1), bias_regularizer=reg(l2=0.2), name='logit'))
    #feature_extractor = tf.keras.Model(
     #   inputs=model.inputs,
      #  outputs=model.get_layer(name='logit').output)



    #print(model.summary)
    #prob = tf.nn.sigmoid_cross_entropy_with_logits(label, logits, name='prob')

    #reward, reward_pred = single_class_split(label, prob[:, 0, 0])
    #_, logits_pred = single_class_split(label, logits[:, 0, 0])
    #weights = tf.abs(0.6 - reward)  # 1.0 or tf.abs(0.75 - reward) or label[:, 2]
    #loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=reward, logits=logits_pred, weights=weights)
    #loss_reg = loss + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    #accuracy = tf_accuracy(reward, reward_pred)
    #precision = tf_precision(reward, reward_pred)
    #recall = tf_recall(reward, reward_pred)
    #f1 = tf_f1(precision, recall, beta=0.5)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryCrossentropy()])
    return model

# Model training function
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
    print(history.history.keys())
    
    hist = pd.DataFrame(history.history)
    loss = hist['loss']
    rbce = hist['loss']
    vloss = hist['loss']
    vbce = hist['loss']
    acc = history.history['loss']

    return epochs, loss, rbce, vloss, vbce, acc
# Model training function for old Bayesian Optimisation implementation
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

# Class to build CNN model
class Placing:
    def __init__(self):
        self.image_shape = {
            'ed': (None, None, 1),
            'rd': (None, None, 1),
            'rc': (None, None, 3),
        }
        self.z_size = 48


         
         
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

    
    @staticmethod
    def binary_decision(string: str, p: float) -> bool:
        return float(int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16) % 2**16) / 2**16 < p

    
class haltCallback(tk.callbacks.Callback):
    def stopping_cond(self, epoch, logs={}):
        if(epoch==2):
            print("Stopping the training")
            self.model.stop_training=True
    
