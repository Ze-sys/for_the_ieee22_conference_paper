
import streamlit as st

logo = 'autoencoders_icon.png'

st.set_page_config(page_title="Aoutoencoder", page_icon=logo, layout="wide")

preamble = f""" <h2  style=""> <img src="https://theaisummer.com/static/6f5e6b0110a7231a9f70cdf0df9190a0
/f4094/topic-autoencoders.png" width=35  height=35 />  Autoencoder  Models Builder </h2> 
"""

# <h5 style="">
# A model builder for  anomaly detection. 
# </h5>

# ### Workflow:
# <h5 style="color:white">
# 1.  Examine and understand the data 🔍 <br>
# 2.  Build data input pipelines 👷‍♀️👷👷‍♂️   <br> 
# 3.  Build multiple models 👷👷‍♂️👷👷‍♂️  <br>
# 4.  Train the models 🏋️‍♂️ 🏋️‍♀️ 🏋️‍♂️ 🏋️‍♀️ <br>
# 5.  Examine models' history (integrate tensorboard) 🔍😍😍😍🔍 <br>
# 6.  Test the models 😋😋😋<br>
# 7.  Select the best and promising model and improve it further 🏆🏆🏆<br>
# 8.  Deploy the winning model 🚀🚀🚀🚀🚀🚀<br>
# </h5>

# - As a demo, I have used the Range Cell (RC) Images from the WERA RADAR (data collected by Ocean Networks Canada).

# """
st.markdown(preamble, unsafe_allow_html=True)
# ## Import TensorFlow and other libraries
import re
import os
import glob
import shutil
import pathlib
import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
# from streamlit_tensorboard import st_tensorboard
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
tf.get_logger().setLevel('ERROR')
todays_date = dt.datetime.today().strftime('%Y-%m-%d:%H:%M:%S')



opts0 = st.sidebar.selectbox('Select Image Type', ['wera_rc_plots', 'spectrograms'])
image_type = opts0 
# ----------------------------------------------------------------------------------------------------------------------
flush_dir_container = st.sidebar.container()
# ----------------------------------------------------------------------------------------------------------------------
image_channel = st.sidebar.selectbox('Select Image Channel', ['RGB', 'Grayscale'])
if image_channel == 'RGB': img_depth = 3 
elif image_channel == 'Grayscale': img_depth = 1
else: st.error('Invalid/unsupported image channel')


arch = st.sidebar.selectbox('Select Architecture Type', ['fully_connected', 'convolutional'])

opts1 = st.sidebar.selectbox('Choose Optimizer', ['Adam', 'SGD'], index=0, key='opt1')

opts2 = st.sidebar.selectbox('Choose Loss Function', ['Mean Squared Error', 'Mean Absolute Error'], index=0, key='opt2')
if opts2 == 'Mean Squared Error':
    opts2 = 'mse'
    loss = losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE, name='mean_squared_error')
elif opts2 == 'Mean Absolute Error':
    opts2 = 'mae'
    loss = losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE, name='mean_absolute_error')

opts3 = st.sidebar.selectbox('Choose Sample Size',
                             ['32', '64', '128', '256', '512', '1024', '2048', '4096', '8192', '16384', '32768'],
                              index=0, key='opt3')

max_img_count = re.sub('[^0-9]', '', opts3)
max_img_count = int(max_img_count)

n_img_2_track = st.sidebar.slider('Number of Images to Track', 1, max_img_count, 1, key='n_img_2_track')

opts4 = st.sidebar.text_input('Enter Max Epochs', '2', key='opt4')

if opts4 is not None:
    max_epochs = int(opts4)
img_track_at_step = st.sidebar.slider('Track Images at Step', 0, max_epochs, 1, key='img_track_at_step')

opts5 = st.sidebar.selectbox('Choose Learning Rate', [0.001, 0.01, 0.1, 1, 10, 100], index=0, key='opt5')
if opts1 is not None and opts5 is not None:
    learning_rate = opts5
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

opts6 = st.sidebar.selectbox('Choose Batch Size', [16, 32, 64, 128, 256, 512, 1024, 2048, 4096], index=0, key='opt6')
if opts6 is not None:
    batch_size = opts6

opts7 = st.sidebar.selectbox('Choose Pixel Sizes', [32, 64, 128, 256, 512], index=0, key='opt7')
if opts7 is not None:
    img_height, img_width = opts7, opts7

opts8 = st.sidebar.selectbox('Choose  threshold %', [75, 80, 85, 90, 95], index=0, key='opt8')
threshold_perc = opts8

more_params_expdr = st.sidebar.checkbox('Show More Parameters', False, key='more_params_expdr')
if more_params_expdr:
    train_test_split = st.sidebar.slider('Train Test Split', 0.0, 1.0, 0.1, key='train_test_split', step=0.1)
    drop_out_rate = st.sidebar.slider('Drop Out Rate', 0.0, 1.0, 0.1, key='drop_out_rate', step=0.1)
    activation = st.sidebar.selectbox('Activation Function', ['relu', 'elu', 'selu', 'tanh', 'sigmoid'], index=0, key='activation')
    if activation == 'relu':
        activation = tf.nn.relu
    elif activation == 'elu':
        activation = tf.nn.elu
    elif activation == 'selu':
        activation = tf.nn.selu
    elif activation == 'tanh':
        activation = tf.nn.tanh
    elif activation == 'sigmoid':
        activation = tf.nn.sigmoid
    kernel_initializer = st.sidebar.selectbox('Kernel Initializer', ['glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal'], index=0, key='kernel_initializer')
    if kernel_initializer == 'glorot_uniform':
        kernel_initializer = tf.keras.initializers.glorot_uniform
    elif kernel_initializer == 'glorot_normal':
        kernel_initializer = tf.keras.initializers.glorot_normal
    elif kernel_initializer == 'he_uniform':
        kernel_initializer = tf.keras.initializers.he_uniform
    elif kernel_initializer == 'he_normal':
        kernel_initializer = tf.keras.initializers.he_normal
   
# ----------------------------------------------------------------------------------------------------------------------
run_model = st.sidebar.button('Run Model')
# ----------------------------------------------------------------------------------------------------------------------

# model_name = "Optimizer:{}_loss:{}_sample_size:{}_epochs:{}_learning_rate:{}_batch_size:{}_pixel_size:{}_by_{}_time_{}".format(
#     opts1, opts2, opts3, opts4, opts5, opts6, opts7, todays_date)

model_name = "Optimizer:{}&loss:{}&samplesize:{}&epochs:{}&learningrate:{}&batchsize:{}&pixelsize:{}&time:{}".format(
    opts1, opts2, opts3, opts4, opts5, opts6, opts7, todays_date)

model_name = re.sub('\s+', '_', model_name)
# for collecting model runs
model_collector = {}

# create logdir path if it does not exist
logdir = 'models/{}/tensorboard_logs/'.format(image_type)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
# shutil.rmtree(logdir, ignore_errors=True) # BE CAREFUL WITH THIS
current_model = "{}{}".format(logdir, model_name)
st.markdown(f'<span style="color:green"> Latest Model: {current_model} </span>', unsafe_allow_html=True)


# write current model name to csv file
def write_current_model_name(threshold_=0):
    with open('models/model_names.csv', 'a') as f:
        # f.write(model_name + '&threshold_perc:'+ threshold_perc +'&threshold_value' + threshold + '\n')
        f.write("{}&threshold_perc:{}&threshold_value:{}".format(model_name, str(threshold_perc), str(threshold_) + '\n'))
# write_current_model_name(threshold)  # to csv file

def save_model(model, model_name,image_type=image_type):
    model.save('{}{}/{}'.format('models/', image_type, model_name))

    st.write('Model Saved')


# read current model name from csv file
def read_saved_model_names():
    if os.path.exists('models/model_names.csv'):
        with open('models/model_names.csv', 'r') as f:
            saved_model_hists = f.read().splitlines()
            cleanup_expdr = st.expander('Cleanup Unselected Models')
            # cleanup_expdr = st.expander('''
            #     Click on buttons inside this expander to remove model(s) history from TensorBoard logs. 
            #     Alternatively, you can flush the log directory by using the button on the sidebar.''')
            # cleanup_expdr.warning("Warning: I won't ask for confirmation!!")
            with cleanup_expdr:
                for model in saved_model_hists:
                    model_hist_to_delete = st.button(model, key=model)
                    if model_hist_to_delete is None:
                        st.warning('WARNING: No model history found on Disk. Please run the model first')
                    else:
                        shutil.rmtree(f'models/{image_type}/tensorboard_logs/' + model, ignore_errors=True)
                        saved_model_hists.remove(model)
                        with open(f'models/{image_type}/model_names.csv', 'w') as f:
                            for updated_list in saved_model_hists:
                                f.write(updated_list + '\n')
                        st.success('Model history deleted from TensorBoard logs on Disk')

                    return saved_model_hists, model_hist_to_delete
    else:
        os.system('touch models/model_names.csv')


read_saved_model_names()  # reads, writes, and deletes model history from TensorBoard logs on Disk as needed


# # Data
# @st.cache
def load_data(data_path='/data/wera_rc_plots/trimmed/', good_image_path='good/',
              bad_image_path='bad/'):
    image_paths_normal, saved_location_normal = glob.glob(data_path + '*.gif'), data_path + good_image_path
    image_paths_anomalous, saved_location_anomalous = glob.glob(data_path + '*.gif'), data_path + bad_image_path
    return image_paths_normal, image_paths_anomalous, saved_location_normal, saved_location_anomalous

def get_pixel_size():
    return img_height, img_width


def get_callbacks(name, logdir):
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=200),
        tf.keras.callbacks.TensorBoard("{}{}".format(logdir, name), write_images=True, write_graph=True)
    ]


def compile_and_fit(model, training_set, validation_set, model_name_, logdir_, optimizer_=optimizer,
                    max_epochs_=max_epochs, img_depth=3, batch_size_=batch_size, loss_=loss):
    img_height_, img_width_ = get_pixel_size()
    model.compile(optimizer=optimizer_, loss=loss_, metrics=['accuracy']),
    model.build(input_shape=(None, img_height_, img_width_, img_depth))

    arch_expdr = st.expander('Model Architecture')
    with arch_expdr:
        cols = st.columns(2)
        with cols[0]:
            st.subheader('Encoder Layer Summary')
            model.encoder.summary(print_fn=lambda x: st.text(x))  # this is to override the default print function
        with cols[1]:
            st.subheader('Decoder Layer Summary')
            model.decoder.summary(print_fn=lambda x: st.text(x))  # this is to override the default print function

    history = model.fit(training_set, training_set,
                        batch_size=batch_size_,
                        epochs=max_epochs_,
                        shuffle=True,
                        callbacks=get_callbacks(model_name_, logdir_),
                        validation_data=(validation_set, validation_set),
                        verbose=0
                        )

    return history, model


def show_images_plotly(images_, decoded_images, label, figure_location):
    '''
    Show  example images in a grid
    '''
    n = 3  # int(min(10,frac*batch_size)) # randomly select n images to display
    original_images_container = st.container()
    reconstructed_images_container = st.container()
    img_cols = st.columns(n)
        
    for i in range(n):

        with original_images_container:
                fg = px.imshow(images_[i, :, :])
                fg.update_layout(width=250, height=250, title='Original {} #{}'.format(label,i), title_x=0.5,
                                margin=dict(l=0, r=0, t=30, b=0), font=dict(size=10))
                img_cols[i].plotly_chart(fg, use_container_width=True)

        with reconstructed_images_container:
                fg = px.imshow(decoded_images[i, :, :])
                fg.update_layout(width=250, height=250, title='Reconstructed {} #{}'.format(label,i), title_x=0.5,
                                margin=dict(l=0, r=0, t=30, b=0), font=dict(size=10, color='gray'))
                img_cols[i].plotly_chart(fg, use_container_width=True)
    return


if run_model:  # The model does not run until I say so.
    if os.uname()[1].split('-')[0] != 'zelalem':
        st.info('To run this model, you need to use the app locally. You can pull the docker image from DockerHub and comment out the st.stop() line to run it locally.')
        st.info('Please see the README file on the Github repo for more details.')
        st.stop()

    if arch == 'convolutional': # not fully implemented yet

        latent_dim = 8
        # I will let the user add more options such as the number of layers, the number of filters, etc.
        class Autoencoder(Model):
            def __init__(self):
                super(Autoencoder, self).__init__()
                self.encoder = tf.keras.Sequential([
                    layers.InputLayer((img_height, img_width, 3)),
                    layers.Conv2D(img_height, kernel_size=(3, 3), strides=1, activation='relu'),
                    layers.Dropout(0.2),
                    layers.Conv2D(.5*img_height, kernel_size=(3, 3), strides=2, activation='relu'),
                    # layers.MaxPooling2D(pool_size=(2,2)),
                    layers.Conv2D(.125*img_height, kernel_size=(3, 3), strides=2, activation='relu'),
                    # layers.MaxPooling2D(pool_size=(2, 2)),

                
                ])
                self.decoder = tf.keras.Sequential([
                    layers.Conv2DTranspose(.125*img_height, kernel_size=(3, 3), strides=2, padding='valid', activation='relu'),
                    layers.Conv2DTranspose(.5*img_height, kernel_size=(3, 3), strides=2, padding='valid', activation='relu'),
                    layers.Conv2DTranspose(img_height, kernel_size=(3, 3), strides=1, padding='valid', activation='relu'),
                    layers.Conv2DTranspose(3, kernel_size=(3, 3), strides=1, padding='valid', activation=None)
                ])

            def call(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded


        autoencoder = Autoencoder()

    #-----------------------------------------------------FC LAYERS--------------------------------------------------------
    elif arch == 'fully_connected':
            
        latent_dim = 8  # dimensionality of the encoding space
        class Autoencoder(Model):
            def __init__(self, latent_dim, img_height, img_width, img_depth):
                super(Autoencoder, self).__init__()
                self.latent_dim = latent_dim
                self.img_height = img_height
                self.img_width = img_width
                self.img_depth = img_depth       
                self.encoder = tf.keras.Sequential([
                layers.Flatten(),
                layers.Dense(8.0*latent_dim, activation='relu'),
                layers.Dense(4.0*latent_dim, activation='relu'),
                layers.Dropout(0.1),                             # this has not been used for runs until Aug 07 2022,00:14PDT, was .1 until OO:27 PDT
                layers.Dense(2.0*latent_dim, activation='relu'),
                layers.Dense(latent_dim, activation='relu'),


                ])
                self.decoder = tf.keras.Sequential([
                layers.Dense(2.0*latent_dim, activation='relu'),
                layers.Dense(4.0*latent_dim, activation='relu'),
                layers.Dropout(0.1),                            # this has not been used for runs until Aug 07 2022, 00:14PDT, was .1 until OO:27 PDT. Reverting back to .1 as of Aug 14, 00:36PDT
                layers.Dense(8.0*latent_dim, activation='relu'),  # this has not been used for runs until Aug 06 2022, 23:58PDT
                layers.Dense(self.img_height * self.img_width * self.img_depth, activation=None),  # as of Aug 14, 00:36PDT, this is the last layer uses no activation function!!
                layers.Reshape((self.img_height, self.img_width, self.img_depth))
                ])

            def call(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        autoencoder = Autoencoder(latent_dim, img_height, img_width, img_depth)
    #----------------------------------------------------FC LAYERS ENDS-----------------------------------------------------


    # the predictions
    def predict(model, data, threshold_, plot_histogram=True):
        from math import prod
        sample_size = data.shape[0]
        total_dims = prod(x_test.shape[1:])
        reconstructions_ = model.predict(data)
        mse_ = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE, name='mean_squared_error')
        loss_ = mse_(reconstructions_.reshape(sample_size, total_dims), data.reshape(sample_size, total_dims))

        # Choose a threshold value that is one standard deviations above the mean.
        print("Predefined Threshold: ", threshold_)
        print('Number of samples reconstructed/predicted: ', sample_size)
        if plot_histogram:
            plt.hist(loss_.numpy(), bins='auto', alpha=0.5, color='blue', density=True)
            plt.vlines(threshold_, 0, 1300, colors='r', linestyles='dashed', linewidth=3)
            plt.xlim(0, 0.025)
            plt.ylim(0, 1200)
            plt.xlabel("Loss")
            plt.ylabel("Density")
            plt.show()
        return loss_.numpy() < threshold_


    def print_stats(predictions, labels, reconstruction_type, tag='', max_epochs_=max_epochs):
        if max_epochs_ < 10:
            return st.write('''
                Not enough loss values to compute statistics (run model with a minimum of 10 epochs. 
                Use at least 30 epochs for more meaningful stats.)
                ''')
        else:
            if tag == 'anomalous':
                # abandoned the two below to avoid confusion
                # accuracy = 1 - accuracy_score(predictions, labels).round(3)  # accuracy_score of correctly predicting anomaly
                # precision = 1 - precision_score(predictions, labels).round(3)  # precision_score of correctly predicting anomaly

                accuracy = accuracy_score(predictions, labels).round(3)  # accuracy_score of correctly predicting anomaly
                precision = precision_score(predictions, labels).round(3)  # precision_score of correctly predicting anomaly
            else:
                accuracy = accuracy_score(predictions, labels).round(3)  # accuracy_score of correctly predicting normal
                precision = precision_score(predictions, labels).round(3)  # precision_score of correctly predicting normal

            st.write("{} Accuracy = {}, Precision = {}, Recall = {}".format(
                reconstruction_type, accuracy, precision,
                recall_score(labels, predictions).round(3)
            ))

            # ## Load the dataset

    if image_type == 'wera_rc_plots':
        image_paths_good, image_paths_bad, saved_location_good, saved_location_bad = load_data()
    elif image_type == 'spectrograms':    
        image_paths_good, image_paths_bad, saved_location_good, saved_location_bad =load_data(data_path='./data/spectrograms/', 
                    good_image_path='SOGC/201910_trimmed/2019/201910/trimmed/small_files/',
                    bad_image_path='bad_spectrogram_examples/trimmed/bad/')
    else:
        raise ValueError('image_type must be either "wera_rc_plots" or "spectrograms"')


    if image_type == 'wera_rc_plots': ext = '.gif'
    elif image_type == 'spectrograms': ext = '.png'

    data_dir = pathlib.Path(__file__).parent.parent.as_posix() + saved_location_good
    data_dir_temp = pathlib.Path(__file__).parent.parent.as_posix() + data_dir + 'temp' # temporary directory to store images
    data_dir = pathlib.Path(data_dir)
    training_images = list(data_dir.glob('*' + ext))
    #----------------------------------------------------subset training DATA  ENDS to fit in memory-----------------------------------------------------
    # copy max_img_count images from good_image_path to a temporary data_dir
    training_images = training_images[:max_img_count]
    if not os.path.exists(data_dir_temp):
        os.makedirs(data_dir_temp)
    for image in training_images:
        shutil.copy(image, data_dir_temp)
    data_dir = pathlib.Path(data_dir_temp)
   #----------------------------------------------------subset training DATA  ENDS to fit in memory ENDS-----------------------------------------------------
    bad_examples = pathlib.Path(__file__).parent.parent.as_posix() + saved_location_bad
    bad_examples = pathlib.Path(bad_examples)
    image_count = len(list(data_dir.glob(ext)))
    print('number of good  images: ', image_count)

    good = list(data_dir.glob(ext))
    bad = list(bad_examples.glob(ext))

    # training set
    good_samples_dir = data_dir
    val_ds_frac = 0.1
    train_ds = tf.keras.utils.image_dataset_from_directory(
        good_samples_dir,
        validation_split=val_ds_frac,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels=None)
    # validation set
    val_ds = tf.keras.utils.image_dataset_from_directory(
        good_samples_dir,
        validation_split=val_ds_frac,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels=None)

    # performance configuration
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
    x_train = [img.numpy().astype("uint8") for img in train_ds.take(1)]
    x_train = np.array(x_train)[0]
    # x_train_reshaped = x_train.reshape(batch_size, img_height, img_width, 3)

    x_test = [img.numpy().astype("uint8") for img in val_ds.take(1)]
    x_test = np.array(x_test)[0]
    # x_test_reshaped = x_test.reshape(batch_size, img_height, img_width, 3)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.


    # ## Compile and Train the model - correctly subsampled the dataset based on user input max_img_count. See how I copy the images to a temporary directory
    model_collector[model_name], trained_model = compile_and_fit(
        # if max_img_count>x_test.shape[0]: max_img_count = x_test.shape[0]
        autoencoder, x_train, x_test, model_name, logdir,
        img_depth=3, batch_size_=batch_size, loss_=loss, optimizer_=optimizer,
        max_epochs_=max_epochs)


    # Create a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
        # Store images to visualize the model's progress.
        images = np.reshape(x_train[0:n_img_2_track], (-1, img_height, img_width, 3))
        tf.summary.image(f"training image example(s)", images, max_outputs=n_img_2_track,
                         step=img_track_at_step)  # at last epoch

    # store graph
    tf.summary.trace_on(graph=True, profiler=True)
    with file_writer.as_default():
        tf.summary.trace_export(
            name="model_trace",
            step=0,
            profiler_outdir=logdir)


  

    # test model for ten good images

    # ### Now that the model is trained, let's check the reconstructions
    fg_cols = st.columns([1, 1, 2])
    encoded_imgs = autoencoder.encoder(x_train).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    
    img_expdr = st.expander('Show example image reconstructions')
    with img_expdr: 
     st.markdown('<hr></hr>', unsafe_allow_html=True)        
     show_images_plotly(x_train, decoded_imgs, 'training image', fg_cols)

    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    with img_expdr:
        st.markdown('<hr></hr>', unsafe_allow_html=True)        
        show_images_plotly(x_test, decoded_imgs, 'validation image', fg_cols)

    # with fg_cols[2]:
        # st.subheader('Tensorboard')
    #     if image_type == 'spectrograms':
    #         st_tensorboard(logdir=logdir, port=port1, width=1750, height=1250)
    #     elif image_type == 'wera_rc_plots':
    #         st_tensorboard(logdir=logdir, port=port2, width=1750, height=1250)

    # Since I am not including tensorboard when deploying, I will just plot loss and accuracy in its place.
     
    reconstructions = autoencoder.predict(x_train)
    reconstructions_v = autoencoder.predict(x_test) # for validation set reconstruction loss calculation

    #  track reconstruction progress- train set
    with file_writer.as_default():
        images = np.reshape(reconstructions[0:n_img_2_track], (-1, img_height, img_width, 3))
        tf.summary.image(f"training image reconstruction example(s)", images, max_outputs=n_img_2_track,
                         step=img_track_at_step)
    # ------------------------------------------------------------------------------------------------------------------
    #
    # Expected keys are "('auto', 'none', 'sum', 'sum_over_batch_size')"
    # but only option none is returning non-empty results at the moment
    mse = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE, name='mean_squared_error')

    l_mse = mse(reconstructions.reshape(len(x_train), img_height * img_width * 3),
                x_train.reshape(len(x_train), img_height * img_width * 3))

    l_mse_v = mse(reconstructions_v.reshape(len(x_test), img_height * img_width * 3),
            x_test.reshape(len(x_test), img_height * img_width * 3))

    # Choose a threshold value that is one standard deviations above the mean.
    threshold = np.percentile(l_mse, [
        threshold_perc])
    # 90th percentile of the losses for the good data. Images with losses above this threshold are considered anomalous
    prediction_batch_size = 100000  # random big number so that we can use all bad examples for reconstruction
    # load the bad images
    bad_samples_dir = bad_examples
    img_height, img_width = img_height, img_width  # same as above
    anomalous_test_data_frac = .05
    anomalous_test_data = tf.keras.utils.image_dataset_from_directory(
        bad_samples_dir,
        # validation_split=anomalous_test_data_frac,
        # subset="training",
        validation_split=None,  # since Aug 23 19:29
        subset=None,   # since Aug 23 19:29
        seed=123,
        image_size=(img_height, img_width),
        batch_size=prediction_batch_size,  # to use all images
        labels=None)

    # performance configuration
    AUTOTUNE = tf.data.AUTOTUNE
    anomalous_test_data = anomalous_test_data.cache().shuffle(buffer_size=prediction_batch_size).prefetch(
        buffer_size=AUTOTUNE)

    y_train = [img.numpy().astype("uint8") for img in anomalous_test_data.take(1)]
    y_train = np.array(y_train)[0]
    y_train = y_train.astype('float32') / 255.

    anomalous_test_data_reconstructions = autoencoder.predict(y_train)
    anomalous_test_data_loss = mse(anomalous_test_data_reconstructions.reshape(
        len(y_train), img_height * img_width * 3), y_train.reshape(
        len(y_train), img_height * img_width * 3))
    #  track reconstruction progress- for anomalous (test) set
    with file_writer.as_default():
        images = np.reshape(y_train[0:n_img_2_track], (-1, img_height, img_width, 3))
        tf.summary.image(f"anomalous image example(s)", images, max_outputs=n_img_2_track,
                         step=img_track_at_step)
        images = np.reshape(anomalous_test_data_reconstructions[0:n_img_2_track], (-1, img_height, img_width, 3))
        tf.summary.image(f"anomalous image reconstruction example(s)", images,
                         max_outputs=n_img_2_track, step=img_track_at_step)

    good_data_loss = l_mse.numpy()
    good_data_loss_v = l_mse_v.numpy()  # validation set loss
    bad_data_loss = anomalous_test_data_loss.numpy()

 # SAVE all losses  for later use ( plotting and analysis)
    lossdir = logdir + '/losses'
    if not os.path.exists(lossdir):
            os.makedirs(lossdir)

    df_l_mse = pd.DataFrame({"loss": good_data_loss})
    df_loss_v = pd.DataFrame({"loss_v": good_data_loss_v})
    df_test_loss = pd.DataFrame({"anomalous_test_data_loss": bad_data_loss})
    df_losses = pd.concat([df_l_mse, df_loss_v, df_test_loss], axis=1)
    losses_file_name = "{}{}/{}&threshold_perc:{}&threshold_value:{}_losses.csv".format(logdir,'losses',model_name, str(threshold_perc), str(threshold[0].round(6)))
    df_losses.to_csv(losses_file_name, index=False)
  
  # show loss and accuracy

    def show_loss_and_accuracy_curves(history_):
        # subplots
        fig = px.line(history_, x=range(len(history_['loss'])), y=['loss', 'accuracy', 'val_loss', 'val_accuracy'] )

        fig.update_layout(
            hovermode="x unified",
            width=800,
            height=600,
            # title="Loss vs Epochs",
            xaxis_title="Epochs",
            yaxis_title="Loss (MSE)",
            legend_title="Loss",
            font=dict(
                family="Times New Roman",
                size=24,
            ),
            margin=dict(
                        l=10,
                        r=10,
                        b=50,
                        t=10,
                        pad=0          
                        ),
            legend=dict(
                        title= "",
                        x=0,
                        y=-.2,
                        bgcolor=None,
                        orientation="h",
                        )
        )
        return fig    



    #------new version---------

    fgcnt = 0
    barmod='group'
    font_fam = "Times New Roman"
    y_ax_lim, histnorm_type = 75, 'percent'
    x_ax_range = [0, .5]        

    if histnorm_type =='count':
        histnorm_type='' 
        ttl='Count'
    else: 
        ttl = histnorm_type

    fig = go.Figure()

    fig.add_trace(go.Histogram(x=df_losses['loss'], name=f'Losses (Normal Training Set Images, n = {len(df_losses["loss"])})',
                                nbinsx=90, histnorm=histnorm_type, opacity=1.0,marker=dict(color='blue')))
    fig.add_trace(go.Histogram(x=df_losses['loss_v'], name=f'Losses (Normal Validation Set Images, n = {len(df_losses["loss_v"].dropna())})', # drop nans from the shorter validation set
                                nbinsx=90, histnorm=histnorm_type, opacity=1.0,marker=dict(color='lightgreen')))
    fig.add_trace(go.Histogram(x=df_losses['anomalous_test_data_loss'], name=f'Losses (Anomalous Images, n = {len(df_losses["anomalous_test_data_loss"].dropna())})',  # drop nans from the shorter test set
                                nbinsx=90, histnorm=histnorm_type, opacity=1.0,marker=dict(color='red')))

    fig.add_trace(go.Scatter(x=[threshold[0], threshold[0]],
                                y=[0, y_ax_lim], mode='lines', marker=dict(size=10),line=dict(color='gray', width=3,
                                dash='dash'), name=f'Threshold ({threshold_perc}th perc. of Normal Image Losses)'))

    fig.update_layout(
        barmode=barmod,
        width=800,
        height=600,
        font=dict(
            family=font_fam,
            size=24),
            
        title=go.layout.Title(
            # text=f"Distribution of Reconstruction Losses",
            xref="paper",
            x=0,
        ),

        xaxis=go.layout.XAxis(
            range=x_ax_range,
            title=go.layout.xaxis.Title(
                text="Loss (mse)",
                font=dict(
                    family=font_fam,
                    size=24,
                    # color="black"
                )
            )
        ),
        yaxis=go.layout.YAxis(
            range=[0,y_ax_lim],
            title=go.layout.yaxis.Title(
                text='% of images',
                font=dict(
                    family=font_fam,
                    size=24,
                    # color="black"
                )
            )
        )
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            x=0,
            y=-.2,
            traceorder="normal",
            font=dict(
                family=font_fam,
                size=18,
                # color="black"
            ),
            bgcolor=None),
            margin=dict(
                        l=10,
                        r=10,
                        b=50,
                        t=10,
                        pad=0          
                        ),
    font = dict(size=24)
    )

#------new version---------
    losfig = show_loss_and_accuracy_curves(model_collector[model_name].history)
    loss_container = st.columns(2)
    loss_container[0].plotly_chart(losfig, use_container_width=True)
    loss_container[1].plotly_chart(fig, use_container_width=True)
    loss_container[1].markdown(f"<h6>The {threshold_perc}th perc. Threshold: {threshold[0].round(4)}</h6>", unsafe_allow_html=True)

    metrics_header = f'''Aggregate Metrics for the Latest Model with the Threshold of {threshold_perc}th perc. of normal images losses:'''

    st.markdown(f'''<h5 style="" > {metrics_header} </h5>''', unsafe_allow_html=True)

    st.markdown('''<h6 style="" >
               The autoencoder model is used to reconstruct the three sets of images (training set- model seen these,
               validation set- not used in back propagation, test set- unseen by the model (all of them are manually 
              checked to be anomalous). 
              All images are tested  against the null hypothesis that they are normal. Since the model is trained to reconstruct only normal images, low accuracy and recall are  
              expected for the anomalous set ( for example, an accuracy of of 0.1 in predicting an anomalous image  as normal is the same as 0.9 accuracy of predicting anomalous as 
              anomalous. 
          </h6>''', unsafe_allow_html=True)

    # print accuracy, recall aand precision for x_train prediction for smaller_threshold (90% accuracy expected)
    preds = predict(autoencoder, x_train, threshold, plot_histogram=False)
    # Since we know all the test images are normal, we can label them such that the loss is less than the threshold
    # (i.e use np.ones() so that MOST of them return bool True)
    test_labels = np.ones(preds.shape, dtype=bool)
    # check the accuracy, precision and recall of the  model
    print_stats(preds, test_labels, 'Training set reconstruction metrics:')

    preds = predict(autoencoder, x_test, threshold, plot_histogram=False)
    # Since we know all the test images are normal, we can label them such that the loss is less than the threshold
    # (i.e use np.ones() so that MOST of them return bool True)
    test_labels = np.ones(preds.shape, dtype=bool)
    # check the accuracy, precision and recall of the  model
    print_stats(preds, test_labels, 'Validation set reconstruction metrics:')

    preds = predict(autoencoder, y_train, threshold, plot_histogram=False)
    #  The null hypothesis is that the model is not able to reconstruct the test data. We label the test image as
    #  anomalous if the loss is greater than the threshold
    #  Under the null hypothesis, we expect very low accuracy and high recall for the test set.
    test_labels = np.ones(preds.shape, dtype=bool)
    # check the accuracy, precision and recall of the  model
    print_stats(preds, test_labels, 'Anomalous (test) set reconstruction metrics:', tag='anomalous')

    print(logdir)
    write_current_model_name(threshold=threshold[0].round(6))  # to csv file

    
    assert autoencoder is not None, "Current model is empty. Run the model to save the the weights."
    save_model(autoencoder,
        "{}&threshold_perc:{}&threshold_value:{}".format(model_name, str(threshold_perc), str(threshold[0].round(6))), image_type)

    st.success("Saving the model weights to disk... This may take a while.")

    # clean up the log directory
    # now that the data are loaded, we can delete the temporary data_dir
    shutil.rmtree(data_dir)



def logdir_flusher(logdir_path):
    assert os.path.exists(logdir_path), f"path {logdir_path} does not exist."
    parent_dir = os.path.dirname(os.path.dirname(logdir_path))
    assert os.path.exists(parent_dir), f"path {parent_dir} does not exist."
    shutil.rmtree(parent_dir, ignore_errors=True)  # remove only last_dir as in path "models/tensorboard_logs/last_dir/""
    flush_dir_container.info("Log directory {} flushed.".format(parent_dir))


with flush_dir_container:
    flush_logdir = st.button('Flush Log Directory')

if flush_logdir:
    if os.uname()[1].split('-')[0] != 'zelalem':
        st.info("Sorry, this is disabled on purpose, unless you are running this app on your own machine. You can pull the docker image from DockerHub and comment out the st.stop() line to run it locally")
        st.info('Please see the README file on the Github repo for more details.')
        st.stop()
    else:
        logdir_flusher(logdir)


    # confirm_flush = flush_dir_container.button("Confirm flushing the log directory")
    # if confirm_flush:  # Not working right now
   

