import os
import glob
import random
import pathlib
import datetime
import pandas as pd
import numpy as np 
from math import prod
import streamlit as st
import tensorflow as tf
import PIL.Image as Image
import plotly.graph_objects as go
from tensorflow.python.ops.numpy_ops import np_config
from sklearn.metrics import accuracy_score, precision_score, recall_score
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
np_config.enable_numpy_behavior()  


logo = 'autoencoders_icon.png'

st.set_page_config(layout="wide",  page_title='Aoutoencoder Models Tester', page_icon=logo)
preamble = f""" <h1  style=""> <img src="https://theaisummer.com/static/6f5e6b0110a7231a9f70cdf0df9190a0
/f4094/topic-autoencoders.png" width=35  height=35 />  Aoutoencoder Models Tester </h1> """
st.markdown(preamble, unsafe_allow_html=True)

st.write ("This app is for interactive testing of autoencoder models.")
st.markdown (
    """

""", unsafe_allow_html=True
)

# image source selection
image_type = st.sidebar.selectbox('Select Image Type', ['wera_rc_plots', 'spectrograms'])
img_list_file =f'./{image_type}/' + image_type + '_image_file_paths.csv'

if image_type == 'wera_rc_plots':
    image_file_extension = '*.gif'
    test_data_options=['unknown (default)', 'known good', 'known bad', 'good and bad','manually labeled good']
elif image_type == 'spectrograms': 
    image_file_extension = '*.png'
    test_data_options=['unknown (default)', 'known good', 'known bad', 'good and bad','manually labeled good','ship','rain']



run_tests = st.sidebar.button('Run Tests')

st.sidebar.markdown("""---""")
st.sidebar.markdown("""App layout Options, credit: [Pablo Fonseca](https://github.com/PablocFonseca/streamlit-aggrid)""") 
# AgGrid settings

grid_height = st.sidebar.number_input("Grid height", min_value=200, max_value=800, value=300)
return_mode = st.sidebar.selectbox("Return Mode", list(DataReturnMode.__members__), index=1)
return_mode_value = DataReturnMode.__members__[return_mode]
update_mode = st.sidebar.selectbox("Update Mode", list(GridUpdateMode.__members__), index=len(GridUpdateMode.__members__)-1)
update_mode_value = GridUpdateMode.__members__[update_mode]
#enterprise modules
enable_enterprise_modules = st.sidebar.checkbox("Enable Enterprise Modules")
if enable_enterprise_modules:
    enable_sidebar =st.sidebar.checkbox("Enable grid sidebar", value=False)
else:
    enable_sidebar = False

#features
fit_columns_on_grid_load = st.sidebar.checkbox("Fit Grid Columns on Load")

enable_selection=st.sidebar.checkbox("Enable row selection", value=True)
if enable_selection:
    st.sidebar.subheader("Selection options")
    selection_mode = st.sidebar.radio("Selection Mode", ['single','multiple'], index=1)

    use_checkbox = st.sidebar.checkbox("Use check box for selection", value=True)
    if use_checkbox:
        groupSelectsChildren = st.sidebar.checkbox("Group checkbox select children", value=True)
        groupSelectsFiltered = st.sidebar.checkbox("Group checkbox includes filtered", value=True)

    if ((selection_mode == 'multiple') & (not use_checkbox)):
        rowMultiSelectWithClick = st.sidebar.checkbox("Multiselect with click (instead of holding CTRL)", value=False)
        if not rowMultiSelectWithClick:
            suppressRowDeselection = st.sidebar.checkbox("Suppress deselection (while holding CTRL)", value=False)
        else:
            suppressRowDeselection=False

enable_pagination = st.sidebar.checkbox("Enable pagination", value=False)
if enable_pagination:
    st.sidebar.subheader("Pagination options")
    paginationAutoSize = st.sidebar.checkbox("Auto pagination size", value=True)
    if not paginationAutoSize:
        paginationPageSize = st.sidebar.number_input("Page size", value=5, min_value=0, max_value=100)
    st.sidebar.text("___")

#------------end AgGrid settings------------------------------

# if run_tests:


def list_all_models(model_path='./models/', image_type_='spectrograms'):
    model_path = model_path + image_type_ + '/'
    df = pd.DataFrame(columns=['model_name','Optimizer', 'loss', 'samplesize', 'epochs', 'learningrate', 'batchsize', 'pixelsize', 'threshold_perc', 'threshold_value'])
    spectrogram_models = glob.glob(model_path + 'Optimizer*')
    timestamps = []
    for sm in spectrogram_models:
        params = sm.split('/')[-1].split('&')
        tt = params[-3].replace('time:','')
        timestamps.append(tt) #datetime.datetime.strptime(tt, '%Y-%m-%d:%H:%M:%S')
        keyv  = [s.split(':') for s in params if not s.__contains__('time')]
        keyv = dict(keyv)
        # keyv['model_path'] = sm
        df1 = pd.DataFrame([dict(keyv)],index=[tt])
        df1['model_name'] = os.path.basename(sm)
        # df1['model build time'] = tt
        df = pd.concat([df, df1], axis=0)
    df['model_build_time'] = timestamps
    df['model_build_time'] = df['model_build_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d:%H:%M:%S'))
    df.sort_values(by='model_build_time', axis=0, ascending=False, inplace=True)  # latest model first
    return df, model_path

with st.spinner('Loading Models...'):
    models_list, model_path = list_all_models(image_type_ = image_type)

gb_models = GridOptionsBuilder.from_dataframe(models_list)
gb_models.configure_grid_options(domLayout='normal')
gb_models.configure_column("model_name", headerCheckboxSelection = True)
gb_models.configure_selection('single', use_checkbox=True, pre_selected_rows=[0])
gb_models.configure_pagination(paginationAutoPageSize=True)



gridOptions_models = gb_models.build()
grid_response_models = AgGrid(
        models_list, 
    gridOptions=gridOptions_models,
    height=grid_height, 
    width='100%',
    data_return_mode=return_mode_value, 
    update_mode=update_mode_value,
    fit_columns_on_grid_load=fit_columns_on_grid_load,
    allow_unsafe_jscode=True, 
    enable_enterprise_modules=enable_enterprise_modules
)

smd = grid_response_models['data']
selected_model_ = grid_response_models['selected_rows']
st.markdown(f'<a style=";font-size:26px;border-radius:50%;">Selected Model (default is latest) </a>',
            unsafe_allow_html=True)
selected_model_df = pd.DataFrame(selected_model_)
# Specify image height and width
img_height, img_width = int(selected_model_df['pixelsize'].values[0]), int(selected_model_df['pixelsize'].values[0])

st.write(selected_model_)
# select_model = st.selectbox(label='Select model', options=models_list.model_path.values) #  won't use this anymore

data_select_apriori_cols = st.columns(2)


dataset_selected = data_select_apriori_cols[0].selectbox(label='Select image set to test', options=test_data_options)
label_type=data_select_apriori_cols[1].selectbox(label='Select label to test against (True means all images are expected to be normal/good)', options=['True', 'False'])


@st.cache_data
def load_data(dataset_selected, img_list_file):
    if dataset_selected == 'unknown (default)':
        test_images_path =  pd.read_csv(img_list_file, skipinitialspace=True, usecols=['unknown_images_path'])
        test_images_path = pathlib.Path(test_images_path.values[0][0])
        test_images_file_list = list(test_images_path.glob(image_file_extension))
        test_images_path_file_count = len(test_images_file_list)
        return test_images_path, test_images_path_file_count, test_images_file_list
    elif dataset_selected == 'known good':
        good_images_path = pd.read_csv(img_list_file, skipinitialspace=True, usecols=['good_images_path'])
        good_images_path = pathlib.Path(good_images_path.values[0][0])
        good_images_file_list = list(good_images_path.glob(image_file_extension))
        good_images_path_file_count = len(good_images_file_list)
        return good_images_path, good_images_path_file_count, good_images_file_list

    elif dataset_selected == 'manually labeled good':
        good_images_path_m = pd.read_csv(img_list_file, skipinitialspace=True, usecols=['manually_labeled_good_images_path'])
        good_images_path_m = pathlib.Path(good_images_path_m.values[0][0])
        good_images_file_list_m = list(good_images_path_m.glob(image_file_extension))
        good_images_path_file_count_m = len(good_images_file_list_m)
        return good_images_path_m, good_images_path_file_count_m, good_images_file_list_m

    elif dataset_selected == 'known bad':
        bad_images_path = pd.read_csv(img_list_file, skipinitialspace=True, usecols=['bad_images_path'])
        bad_images_path = pathlib.Path(bad_images_path.values[0][0])
        bad_images_file_list = list(bad_images_path.glob(image_file_extension))
        bad_images_path_file_count = len(bad_images_file_list)
        return bad_images_path, bad_images_path_file_count, bad_images_file_list
    elif dataset_selected == 'good and bad': 
        good_and_bad_path = pd.read_csv(img_list_file, skipinitialspace=True, usecols=['good_and_bad_images_path'])
        good_and_bad_images_path = pathlib.Path(good_and_bad_path.values[0][0])  
        good_and_bad_images_file_list = list(good_and_bad_images_path.glob(image_file_extension))
        good_and_bad_path_file_count = len(good_and_bad_images_file_list)                     
        return good_and_bad_images_path, good_and_bad_path_file_count, good_and_bad_images_file_list

    elif dataset_selected == 'ship':
        ship_images_path = pathlib.Path('./data/spectrograms/ship/trimmed/')  
        ship_images_file_list = list(ship_images_path.glob(image_file_extension))
        ship_images_file_count = len(ship_images_file_list)                     
        return ship_images_path, ship_images_file_count, ship_images_file_list
    elif dataset_selected == 'rain':
        rain_images_path = pathlib.Path('./data/spectrograms/rain/trimmed/')  
        rain_images_file_list = list(rain_images_path.glob(image_file_extension))
        rain_images_file_count = len(rain_images_file_list)                     
        return rain_images_path, rain_images_file_count, rain_images_file_list

    else:
        st.write('No dataset selected')
        return None, None


test_samples_dir, batch_size, test_files_list = load_data(dataset_selected, img_list_file)
# st.write(test_samples_dir)


test_data = tf.keras.utils.image_dataset_from_directory(
    test_samples_dir,
    validation_split=None, #anomalous_test_data_frac,
    subset=None, #"training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels=None)


y_test = [img.numpy().astype("uint8") for img in test_data.take(1)]
y_test = np.array(y_test)[0] / 255.
y_test = tf.cast(y_test, tf.float32)
y_test = tf.image.resize(y_test, (img_height, img_width))

class Predict:
    def __init__(self, selected_model=None, model_path=None, dataset_selected=None):
        self.selected_model = selected_model
        self.dataset_selected = dataset_selected
        self.selected_model_path = model_path + selected_model.model_name.values[0]
    # @st.cache_data   
    def load_model(self):
        self.model = tf.keras.models.load_model(self.selected_model_path)
        return  self.model

    def predict(self,data,plot_histogram=True):
        # load data
        self.data = data
        # load model
        self.model = self.load_model()
        self.sample_size = data.shape[0]
        total_dims = prod(data.shape[1:])
        reconstructions = self.model.predict(self.data) 
        mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE, name='mean_squared_error')
        loss = mse(reconstructions.reshape(self.sample_size, total_dims), data.reshape(self.sample_size, total_dims))
        # save loss values to  csv file
        loss_df = pd.DataFrame(loss.numpy())
        loss_df.to_csv('loss_values_data_type:{}_model:{}.csv'.format(self.dataset_selected,self.selected_model.model_name.values[0]), index=['full_loss'], header=None)
        
        #Choose a threshold value
        st.write("Predefined Threshold: ", self.selected_model.threshold_value.values[0])
        st.write('Number of samples reconstructed/predicted: ', self.sample_size)
        if plot_histogram:
            threshold_perc = self.selected_model.threshold_perc.values[0]
        barmod='group'
        font_fam = "Times New Roman"
        nbinsx = int(len(loss)/2)
        hst= np.histogram(loss.numpy(), bins=nbinsx)
        y_ax_lim, histnorm_type = int(max(hst[0])), 'percent'
        x_ax_range = [0, .5]        
           

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=loss.numpy(), name=f'Reconstruction Loss, Test Images (n = {self.sample_size})',
                                    nbinsx=nbinsx, histnorm=histnorm_type, opacity=1.0,marker=dict(color='blue')))
                             
        # fig.add_trace(go.Histogram(x=df_losses['loss_v'], name=f'Losses (Normal Validation Set Images, n = {len(df_losses["loss_v"].dropna())})', # drop nans from the shorter validation set
        #                             nbinsx=90, histnorm=histnorm_type, opacity=1.0,marker=dict(color='lightgreen')))
        # fig.add_trace(go.Histogram(x=df_losses['anomalous_test_data_loss'], name=f'Losses (Anomalous Images, n = {len(df_losses["anomalous_test_data_loss"].dropna())})',  # drop nans from the shorter test set
        #                             nbinsx=90, histnorm=histnorm_type, opacity=1.0,marker=dict(color='red')))

        fig.add_trace(go.Scatter(x=[self.selected_model.threshold_value.values[0],  self.selected_model.threshold_value.values[0]],
                                    y=[0, y_ax_lim], mode='lines', marker=dict(size=10),line=dict(color='gray', width=3,
                                    dash='dash'), name=f'Threshold ({threshold_perc}th perc. of Normal Image Losses)'))


        fig.update_layout(
            barmode=barmod,
            width=1000,
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
                        size=40,
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
                        size=40,
                        # color="black"
                    )
                )
            )
        )
        fig.update_layout(
            legend=dict(
                x=.75,
                y=.95,
                traceorder="normal",
                font=dict(
                    family=font_fam,
                    size=20,
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
        font = dict(size=34)
        )

        st.plotly_chart(fig, use_container_width=True)

        return loss.numpy() < float(self.selected_model.threshold_value.values[0]), reconstructions # returns a boolean array of True/False. True if the loss is smaller than the threshold.

    def print_stats(self, predictions, labels):
        return st.write(f"Overall metrics for {len(labels)} test images: ", "Accuracy = {}".format(accuracy_score(labels, predictions).round(3)),", Recall = {}".format(recall_score(labels, predictions).round(3)),", Precision = {}".format(precision_score(labels, predictions).round(3)))


st.write('Test data dimension:\n',y_test.shape)

helper = Predict(model_path=model_path, selected_model=selected_model_df, dataset_selected=dataset_selected) 
helper.load_model()  # load model from AGGrid Table

preds, predicted_imgs = helper.predict(y_test)
st.markdown(f'<a style=";font-size:20px;border-radius:50%;">Note: Only when a known set (pre-labeled) of images are tested the metrics below apply. If unknown set of images are selected for model testing, the accuracy simply means the fraction of images the model predicts as normal, given the reconstruction threshold. </a>',
            unsafe_allow_html=True)
# predicted_imgs = predicted_imgs.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], y_test.shape[3])

dfiles = pd.DataFrame({'file_names': [os.path.basename(f_name) for f_name in  test_files_list]})

selected_predicted_img_name = st.selectbox('Select an image to see its reconstruction', dfiles['file_names'])

# index of the selected image in the test set
selected_index = dfiles[dfiles['file_names'] == selected_predicted_img_name].index.values[0]


selected_predicted_imgs = predicted_imgs[selected_index]
selected_test_imgs = y_test.numpy()[selected_index]


orig_pred_expdr = st.expander('Example Input and Predicted Images:')
orig_pred_expdr_cols = st.columns(6)
with orig_pred_expdr:
    # orig_pred_expdr_cols[0].write(f'Original Image: ' + selected_predicted_img_name )
    orig_pred_expdr_cols[0].subheader('A. Origional')
    orig_pred_expdr_cols[0].image(selected_test_imgs, clamp=True, width = 300, caption=selected_predicted_img_name , use_column_width=False)
    #     cols[0].image(y_test.numpy()[i] , width=250, use_column_width=False, caption=fig_label[0] + '\n' + 'Original'  + ' (annotation: {})'.format(get_img_class(dataset_selected)))

    # orig_pred_expdr_cols[1].write('Reconstructed Image' + '\nPredicted label: ' + recon_result)
    orig_pred_expdr_cols[1].subheader('B.  Reconstructed')
    orig_pred_expdr_cols[1].image(selected_predicted_imgs, clamp=True, width=300,  caption=selected_predicted_img_name, use_column_width=False)



# for neater image labeling
def get_label(pred):
    if pred:
        return 'normal'
    else:
        return 'anomalous'

def get_img_class(dataset_selected_):
    if dataset_selected_ == 'unknown (default)':
        img_class = 'unknown'
    elif dataset_selected == 'known good':
        img_class = 'normal'
    elif dataset_selected_ == 'known bad':
        img_class = 'anomalous'
    elif dataset_selected_ == 'good and bad':
        img_class = 'unknown'
    elif dataset_selected_ == 'manually labeled good':
        img_class = 'normal'
    elif dataset_selected_ == 'ship':
        img_class = 'ship'
    elif dataset_selected_ == 'rain':
        img_class = 'rain'
    else:
        img_class = 'unlabeled'
    return img_class


 #---------------------- uncomment for looking more images ends ----------------------


# random_3_idxs = np.random.choice(len(test_files_list), 3, replace=False)
# # random_3_idxs=[16,89,8] #for known bad
# # random_3_idxs=[ 16,8, 630] #for known good
# # random_3_idxs=[408,176,991] #for unknown
# # random_3_idxs = [64,3,35] #for ship
# # random_3_idxs = [15,6,46] #for rain
# st.write('Randomly selected 3 images:')
# st.write(random_3_idxs)

# for i, fig_label in zip(random_3_idxs, [('A)','B)'), ('C)', 'D)'), ('E)', 'F)'), ('G)', 'H)'), ('I)','J)'), ('K)', 'L)'),('M)', 'N)')]):
#     cols = st.columns([3,3,21])
#     # cols[0].markdown(f'<a style="font-size:16px;border-radius:50%;">{fig_label[0]} </a>',
#     #         unsafe_allow_html=True)
#     cols[0].image(y_test.numpy()[i] , width=250, use_column_width=False, caption=fig_label[0] + '\n' + 'Original'  + ' (annotation: {})'.format(get_img_class(dataset_selected)))
#     # cols[1].markdown(f'<a style="font-size:30px;border-radius:90%;">➩</a>',
#     #         unsafe_allow_html=True)

#     # cols[1].markdown(f'<a style="font-size:16px;border-radius:50%;">{fig_label[1]} </a>',
#     #         unsafe_allow_html=True)    
#     cols[1].image(predicted_imgs[i], clamp=True, width=250, use_column_width=False, caption=fig_label[1] + '\n' + 'Predicted'  + ' (predicted label: {})'.format(get_label(preds[i])
# ))
    
 #---------------------- uncomment for looking more images ends ----------------------

# st.write(f'Reconstruction Result: {recon_resut}')
st.write(f'Reconstruction Threshold = {helper.selected_model.threshold_value.values[0]}')
st.write(f'Reconstruction Threshold Percentage = {helper.selected_model.threshold_perc.values[0]}')
# st.write(f'Overall metrics for {len(dfiles)} test images:')

# # selected_predicted_imgs_index = st.selectbox('Select an image to view:', test_file_names)
# with orig_pred_expdr:
#     orig_pred_expdr_cols[0].image(y_test.numpy()[0], caption='Input Image: ' + os.path.basename(test_files_list[0]), use_column_width=False)
#     orig_pred_expdr_cols[1].image(predicted_imgs[0], caption='Predicted Image: ' + os.path.basename(test_files_list[0]), use_column_width=False)


if label_type == 'True':
    test_labels = np.ones(preds.shape, dtype=bool)
else:
    test_labels = np.zeros(preds.shape, dtype=bool)
# test_labels[0:10] = True # to see how precission and recall  change
helper.print_stats(preds, test_labels)

df = pd.DataFrame({'Index':np.array(range(0, preds.shape[0])),'Image': [os.path.basename(x) for x in test_files_list], 
'Labels': test_labels, 'Predictions': preds, "Data Specialist's Note":np.tile(['Click here to add your note.'],len(preds)),
 'Data Specialist':np.tile(['Click here to add your name.'],len(preds)), 'Data Assessment Date':np.tile(['Click here to add date in format YYYY-MM-DD.'],len(preds)), 'Model used':np.tile([helper.selected_model.model_name.values[0]],len(preds))})


def annotation_dir(path = os.getcwd(), up_dir='data/{}/data_annotation/'.format(image_type)):
    return os.path.join(path, up_dir)

data_annotation_dir = annotation_dir()
if not os.path.exists(data_annotation_dir):
    os.makedirs(data_annotation_dir)
annotations_file = os.path.join(data_annotation_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.csv')

gb = GridOptionsBuilder.from_dataframe(df)

cellsytle_jscode = JsCode("""
function(params) {
    if (params.value == false) {
        return {
            'color': 'white',
            'backgroundColor': 'red'
        }
    } else {
        return {
            'color': 'blue',
            'backgroundColor': 'white'
        }
    }
};
""")
gb.configure_column("Predictions", cellStyle=cellsytle_jscode)

if enable_selection:
    gb.configure_selection(selection_mode)
    if use_checkbox:
        gb.configure_selection(selection_mode, use_checkbox=True, groupSelectsChildren=groupSelectsChildren, groupSelectsFiltered=groupSelectsFiltered)
    if ((selection_mode == 'multiple') & (not use_checkbox)):
        gb.configure_selection(selection_mode, use_checkbox=False, rowMultiSelectWithClick=rowMultiSelectWithClick, suppressRowDeselection=suppressRowDeselection)

if enable_pagination:
    if paginationAutoSize:
        gb.configure_pagination(paginationAutoPageSize=True)
    else:
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=paginationPageSize)

gb.configure_grid_options(domLayout='normal')
gb.configure_column("Index", headerCheckboxSelection = True)

def select_random_rows(df, n):
        random.seed(1)
        random_rows = random.sample(range(0, df.shape[0]), n)
        return random_rows
max_pre_selected = 8
randomly_selected_rows = sorted(select_random_rows(df, max_pre_selected))
gb.configure_selection('multiple', pre_selected_rows=randomly_selected_rows)
gb.configure_default_column(editable=True)


st.write("Randomly Selected Rows: {}".format(randomly_selected_rows))

gridOptions = gb.build()
grid_response = AgGrid(
    df, 
    gridOptions=gridOptions,
    height=grid_height, 
    width='100%',
    data_return_mode=return_mode_value, 
    update_mode=update_mode_value,
    fit_columns_on_grid_load=fit_columns_on_grid_load,
    allow_unsafe_jscode=True, 
    enable_enterprise_modules=enable_enterprise_modules
    )

def open_and_show_image(image_index,image_path):
    img = Image.open(image_path)
    st.image(img, width=250, caption='Image Index: ' + str(image_index), use_column_width=False)


df = grid_response['data']
selected = grid_response['selected_rows']
selected_df = pd.DataFrame(selected).apply(pd.to_numeric, errors='coerce')

st.markdown(f'<a style=";font-size:26px;border-radius:50%;">Randomly Selected Images </a>',
            unsafe_allow_html=True)



def markdown_if_true_or_false(value):
    if value:
        return st.markdown(f'<span style="color: green;font-size:18px;border-radius:50%">{value}</span>',
                        unsafe_allow_html=True)
    else:
        return st.markdown(f'<span style="color: red;font-size:18px;border-radius:50%">{value}</span>',
                        unsafe_allow_html=True)

img_cols = st.columns(len(randomly_selected_rows))
for i, n in enumerate(randomly_selected_rows):
    with img_cols[i]:
        # ttl = "Image Index: {}<br> Prediction: {}<br> Tested against: {}".format(n, df['Predictions'][n])
        # st.markdown(f'<a style=";font-size:18px;border-radius:50%;">{ttl}</a>',
        #     unsafe_allow_html=True)
        markdown_if_true_or_false(df['Predictions'][n])

        open_and_show_image(n, test_files_list[n])

# Allow user to select random images show, max 8 images
st.markdown(f'<a style=";font-size:26px;border-radius:50%;">User Selected Images </a>',
            unsafe_allow_html=True)

user_selected_img_row = selected_df.copy()

user_selected_img_row = user_selected_img_row.query('Index != @randomly_selected_rows').reset_index(drop=True)



if user_selected_img_row.shape[0] > 0:
    assert len(user_selected_img_row) <= 8, f"User selected images cannot be more than 8. Deselect at least {len(user_selected_img_row)- 8} images to continue..."
    user_selected_img_cols = st.columns(len(user_selected_img_row))
    for i, n in enumerate(user_selected_img_row.Index):
        with user_selected_img_cols[i]:
            # ttl = "Image Index: {}<br> Prediction: {}<br> Tested against: {}".format(n, df['Predictions'][n])
            # st.markdown(f'<a style=";font-size:18px;border-radius:50%;">{ttl}</a>',
            #     unsafe_allow_html=True)
            markdown_if_true_or_false(df['Predictions'][n])
            open_and_show_image(n, test_files_list[n])

@st.cache_data
def df_to_csv(df):
    return df.to_csv().encode('utf-8')


def csv_downloader(x,annotations_file_=annotations_file, image_type_=image_type):
    st.markdown(f'<a style=";font-size:20px;border-radius:50%;">Download Annotations.</a>',
                unsafe_allow_html=True)
    st.download_button(
        "Download",
        df_to_csv(x),
        "annotations_{}_{}".format(image_type_, os.path.basename(annotations_file_)),
        key='download-csv_format'
    )
       
    # st.write(gridOptions)

annotation_cols = st.columns(10)
with annotation_cols[0]:
    st.markdown(f'<a style=";font-size:20px;border-radius:50%;">Save Annotations.</a>',
            unsafe_allow_html=True)
    if st.button('Save'):
        df.to_csv(annotations_file, index=False) # this needs some more work to save all annotations. Enable edit mode of ag-grid to save all annotations. 
        st.success(f'Annotations saved.')
with annotation_cols[1]:
    csv_downloader(df)



