import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, \
Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import gzip
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report, log_loss
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle
import argparse

def load_mnist(path, is_train=True):
    """ Load MNIST data from `path`

    :param path: str
    :param is_train: bool
    :return: images: np.array (samples,28,28,1)
    :return:  labels: np.array (samples, 10)

    """


    if is_train:
        prefix = 'train'
    else:
        prefix = 't10k'


    labels_path = os.path.join(path,'{}-labels-idx1-ubyte.gz'.format(prefix))
    images_path = os.path.join(path,'{}-images-idx3-ubyte.gz'.format(prefix))

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28,28,1)

    images = images/255.0 #normalize [0,1]
    labels = to_categorical(labels, 10)

    return images, labels


def sampling(args):
    """

    :param mu: a Z_dim vector of floats representing the current mean vector of the latent variable z
    :param log_sigma:  a Z_dim vector of floats (since we assume the co-variance matrix is 0 except the diagonal)
    representing the current log variance of the latent variable z (we use log variance because variance should always be positive
    and we don't want to constraint our network to only positive values, so the log variance will accept any value (also negaives)
    :return: a Z_dim vector sampled from our Gaussian. This is caulated by 𝑍=𝜇+𝜎𝜀 , where  𝜀 ~ N(0, I)
    """
    mu, log_variance = args
    epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
    return mu + K.exp(log_variance / 2) * epsilon


def build_vae_encoder(conv_filters, conv_kernel_size, conv_strides, use_batch_norm=False, use_dropout=False):

    """
    Gets batched of images (batch_size,28,28,1), apply  Conv --> BatchNorm --> Dropout  to downsample
    and use two separate FC layers to learn the mean and log variance (both FC have an output dim of Z_dim)
    the output will be a sample of lengh Z_dim from a gaussian with the aforementioned mean and variance

    :param conv_filters: list of int, number of output channels in each layer
    :param conv_kernel_size:  list of int, kernel size in each layer
    :param conv_strides: list of int, stride length in each layer
    :param use_batch_norm: boolean, use batch norm between layers
    :param use_dropout:  boolean, use dropout between layers
    :return: encoder_input keras.layers.Input: used later to connect to the decoder model
    :return encoder_output keras.layers.Lambda: used to get the Z_dim latent vector
    :return mu: vector in shape (Z_dim,) used for KL loss
    :return log_variance: vector in shape (Z_dim,) used for KL loss
    :return shape_before_flattening: tuple (highet , width, channels) : required for reshaping latent vector while building decoder
    :return encoder_model: keras.model.Model, used to connect to the decoder model

    """

    global K
    K.clear_session()

    # Number of Conv layers
    n_layers = len(conv_filters)

    # Define model input
    encoder_input = Input(shape=(28,28,1), name='encoder_input')
    x = encoder_input


    # Add convolutional layers
    for i in range(n_layers):
        x = Conv2D(filters=conv_filters[i],
                   kernel_size=conv_kernel_size[i],
                   strides=conv_strides[i],
                   padding='same',
                   name='encoder_conv_' + str(i)
                   )(x)
        if use_batch_norm:
            x = BatchNormalization()(x)

        x = LeakyReLU(0.2)(x)

        if use_dropout:
            x = Dropout(rate=0.4)(x)

    shape_before_flattening = K.int_shape(x)[1:]

    x = Flatten()(x)

    mu = Dense(Z_dim, name='mu')(x)
    log_variance = Dense(Z_dim, name='log_var')(x)


    encoder_output = Lambda(sampling, name='encoder_output')([mu, log_variance])

    encoder_model = Model(encoder_input, encoder_output)
    print(encoder_model.summary())

    return encoder_input, encoder_output, mu, log_variance, shape_before_flattening, encoder_model





def build_decoder(shape_before_flattening, conv_filters, conv_kernel_size, conv_strides):

    """
    Gets a Z_dim latent vector and uses inversed convolution to upsample and reconstruct the image

    :param shape_before_flattening: tuple (highet , width, channels)
    :param conv_filters: list of int, number of output channels in each layer
    :param conv_kernel_size:  list of int, kernel size in each layer
    :param conv_strides: list of int, stride length in each layer

    :return decoder_input keras.layers.Input: used later to connect to the encoder model
    :return decoder_output keras.layers.Activation: used to get the reconstructed image
    :return decoder_model: keras.model.Model, used to connect to the encoder model

    """

    # Number of Conv layers
    n_layers = len(conv_filters)

    # Define model input
    decoder_input = Input(shape=(Z_dim,), name='decoder_input')

    # To get an exact mirror image of the encoder
    new_shape = np.prod(shape_before_flattening)  # flatten (highet * width * channels)
    x = Dense(new_shape)(decoder_input)
    x = Reshape(shape_before_flattening)(x)

    # Add convolutional layers
    for i in range(n_layers):
        x = Conv2DTranspose(filters=conv_filters[i],
                            kernel_size=conv_kernel_size[i],
                            strides=conv_strides[i],
                            padding='same',
                            name='decoder_conv_' + str(i)
                            )(x)


        if i < n_layers - 1:
            x = LeakyReLU()(x)
        else:
            x = Activation('sigmoid')(x) # image pixels are between 0 and 1 so reconstructed image is the same

    # Define model output
    decoder_output = x
    decoder_model = Model(decoder_input, decoder_output)
    print(decoder_model.summary())
    return decoder_input, decoder_output, decoder_model





def rmse_loss(y_true, y_pred):
    """

    :param y_true: true image (batch_size, height, width)
    :param y_pred: reconstructed image (batch_size, height, width)
    :return: float: rmse loss across all axis
    """
    return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

def kl_loss(y_true, y_pred):
    """

    :param y_true: true image (batch_size, height, width)
    :param y_pred: reconstructed image (batch_size, height, width)
    :return: float: KL divergence compared to a gaussian with (mu=0, sigma = I),
    this is used to ensure that the encodings are very similar to a multivariate standard normal distribution
    """
    kl_loss = -0.5 * K.sum(1 + log_var - K.square(mean_mu) - K.exp(log_var), axis=1)
    return kl_loss


def total_loss(y_true, y_pred):
    """
    The loss function is a sum of RMSE and KL Divergence. A weight is assigned to the RMSE loss.
    Too High weight means that more emphasis is on RMSE, leading to a regular auto encoder
    (latent space is not continuous, it is more fixed on compressing the given images and not in
    learning a distribution to generate new samples).
    Too Low weight means the quality of the reconstructed images will be poor.

    :param y_true: true image (batch_size, height, width)
    :param y_pred: reconstructed image (batch_size, height, width)
    :return: float
    """
    return LOSS_FACTOR * rmse_loss(y_true, y_pred) + kl_loss(y_true, y_pred)






def plot_compare_vae(batch_images, vae_model):
    """

    :param batch_images: (batch_size, 28,28 ,1 ) of images
    :param vae_model: keras.models.Model
    :return:
    """


    n_to_show = batch_images.shape[0]
    reconst_images = vae_model.predict(batch_images)

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.title('Original Images (top row) vs Re-constructed Images (bottom row)')

    # real images
    for i in range(n_to_show):
        img = batch_images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i + 1)
        sub.axis('off')
        sub.imshow(img)

    # reconstructed
    for i in range(n_to_show):
        img = reconst_images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i + n_to_show + 1)
        sub.axis('off')
        sub.imshow(img)
    plt.savefig('./images/reconstructed_vs_real.png')




def transorm_X_to_Z(batch_images, vae_encoder):
    """

    :param batch_images: (batch_size, 28,28 ,1 ) of images
    :param vae_encoder: keras.models.Model
    :return: transform to latent space (batch_size, Z_dim)
    """

    return vae_encoder.predict(batch_images)



def vae_generate_images(vae_decoder, n_to_show=10):
    """
    uses the trained decoder to generate "n_to_show"  new images from random noise

    :param vae_decoder: keras.models.Model
    :param n_to_show: int
    :return:
    """

    reconst_images = vae_decoder.predict(np.random.normal(0,1,size=(n_to_show,Z_dim)))

    fig = plt.figure(figsize=(15, 3))

    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(n_to_show):
        img = reconst_images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i+1)
        sub.axis('off')
        sub.imshow(img)
    plt.title('New Generated Images')

    plt.savefig('./images/generated.png')

def split_train_test(X, y, size):
    """
    creates a training set of images with size="size", makes sure all labels have equal distribution

    :param X: np.array (60000, 28,28,1)
    :param y: np.array (60000, 10)
    :param size: int how many samples to take to training set
    :return: np.array(size, 28,28,1), np.array(size, ) , np.array (60000-size, 28,28,1), np.array(60000-size,)
    """
    np.random.seed(7)
    y = np.argmax(y, axis=1) # turn one hot to single label
    final_indices = np.array([])
    for cls in range(10):
        indices = np.argwhere(y == cls).reshape(-1) # get all indices that fit the class
        selected_indices = np.random.choice(indices, size, replace=False) # select "size" indices of class
        final_indices = np.concatenate([final_indices, selected_indices]) # append to final indices
    final_indices = final_indices.astype(int)
    X_train, y_train = X[final_indices, :,:,:], y[final_indices] # select x,y with final indices for train
    return X_train, y_train


def train_svm(Zx_train, y_train, Zx_test, y_test):
    """
    trains linear SVM based on latent representation, also does grid search with cross validation and outputs some metrics
    :param Zx_train: np.array (sample_size, Z_dim)
    :param y_train: (sample_size, 1)
    :param Zx_test: (10000-sample_size, Z_dim)
    :param y_test: (10000-sample_size, 1)
    :return: sklearn.svm.SVC
    """

    # with with linear SVM with grid search and 3 fold cross validation
    param_grid =  {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000], 'verbose': [False]}
    #param_grid =  {'kernel': ['rbf'],  'verbose': [False]}

    svm_model = GridSearchCV(SVC(), param_grid, cv=3)
    svm_model.fit(Zx_train, y_train)
    print('Best score for training data:', svm_model.best_score_, "\n")
    print('Best C:', svm_model.best_estimator_.C, "\n")
    print('Best Kernel:', svm_model.best_estimator_.kernel, "\n")
    print('Best Gamma:', svm_model.best_estimator_.gamma, "\n")

    best_model = svm_model.best_estimator_
    y_pred = best_model.predict(Zx_test)
    #evaluate
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    print(classification_report(y_test, y_pred))
    print("Training set score for SVM: %f" % best_model.score(Zx_train, y_train))
    print("Testing  set score for SVM: %f" % best_model.score(Zx_test, y_test))
    return best_model

def args_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-z", "--latent_dim", type=int, default=50)
    parser.add_argument("-e", "--epochs", type= int, default=100)
    parser.add_argument("-b", "--batch_size", type= int, default=128)
    parser.add_argument("-l", "--load",  type=bool, help="pretrained model name")
    return parser.parse_args()


if __name__=="__main__":

    args = args_parsing()

    # Disable eager execution in order to run special loss function. More about eager execution in tf :
    # https://medium.com/coding-blocks/eager-execution-in-tensorflow-a-more-pythonic-way-of-building-models-e461810618c8

    tf.compat.v1.disable_eager_execution()
    #TODO  generate (100,600,1000,3000) (Xi, Yi) pairs from the data and get their (Zi, Yi) pairs. This is the trainig set
    # TODO Test set should be the rest of the (Xi--> Zi, Yi)

    # define hyper-params
    Z_dim = args.latent_dim
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = 0.0005
    LOSS_FACTOR = 10000
    PATH = './data'
    if args.load:
        load_pretrained = True
    else:
        load_pretrained = False
    CHECKPOINT_PATH = './checkpoints/vae_weights.h5'


    X, y = load_mnist(PATH, True)


    # define VAE encoder
    vae_encoder_input, vae_encoder_output, mean_mu, log_var, vae_shape_before_flattening, vae_encoder = \
        build_vae_encoder(
        conv_filters=[32, 64, 128],
        conv_kernel_size=[4, 4, 4],
        conv_strides=[2, 2, 1],
        use_batch_norm=True,
        use_dropout=True)

    # define VAE decoder
    vae_decoder_input, vae_decoder_output, vae_decoder = build_decoder(
        shape_before_flattening=vae_shape_before_flattening,
        conv_filters=[64, 32, 1],
        conv_kernel_size=[4, 4, 4],
        conv_strides=[1, 2, 2])

    # define VAE complete model
    # Input to the combined model will be the input to the encoder.
    # Output of the combined model will be the output of the decoder.
    vae_model = Model(vae_encoder_input, vae_decoder(vae_encoder_output))

    if load_pretrained:
        vae_model.load_weights(CHECKPOINT_PATH)

    # compile
    adam_optimizer = Adam(lr=LEARNING_RATE)
    vae_model.compile(optimizer=adam_optimizer, loss=total_loss, metrics=[rmse_loss, kl_loss])

    # checkpoints
    checkpoint_vae = ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True, verbose=1, save_best_only=True)




    #load model to memory ~ 60 K images (28,28)
    X, y = load_mnist(PATH, is_train=True)

    # fit
    if not load_pretrained:
        vae_model.fit(X, X, epochs=N_EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpoint_vae])

    # take an example batch and plot real images vs re-constructed images
    example_batch = X[np.random.choice(X.shape[0], 10, replace=False), :, :, :]
    plot_compare_vae(example_batch, vae_model)



    # generate new images
    vae_generate_images(vae_decoder, n_to_show=4)


    # split train/test for SVM
    SIZES =[100, 600, 1000, 3000]
    for size in SIZES:
        X_train, y_train = split_train_test(X,y, size)

        # transform X to Z_x

        X_test, y_test = load_mnist(PATH, is_train=False)
        y_test = np.argmax(y_test, axis=1)

        Zx_train = transorm_X_to_Z(X_train, vae_encoder)
        Zx_test = transorm_X_to_Z(X_test, vae_encoder)

        # train/load SVM
        if not load_pretrained:
            print(' Training SVM for size={}'.format(size))
            svm_model = train_svm(Zx_train, y_train, Zx_test, y_test)
            pickle.dump(svm_model, open('./checkpoints/svm_model_size={}.sav'.format(size), 'wb'))
        else:
            print(' Loading pretrained SVM for size={}'.format(size))
            svm_model = pickle.load(open('./checkpoints/svm_model_size={}.sav'.format(size), 'rb'))

        y_pred= svm_model.predict(Zx_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # turn to one-hot
        y_pred = np.eye(10)[y_pred]
        y_test = np.eye(10)[y_test]
        print('Test Set Cross Entropy Result for Training Samples: {} is {}'.format(size, log_loss(y_test, y_pred)))
