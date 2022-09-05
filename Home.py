
import streamlit as st

logo = 'autoencoders_icon.png'

st.set_page_config(page_title="Aoutoencoder", page_icon=logo, layout="wide")

preamble = f""" <h1  style=""> <img src="https://theaisummer.com/static/6f5e6b0110a7231a9f70cdf0df9190a0
/f4094/topic-autoencoders.png" width=35  height=35 />  Autoencoder  Models Builder & Tester </h1> 

<!--
### Workflow:
<h5 style="color:orange">
1.  Examine and understand the data 🔍 <br>
2.  Build data input pipelines 👷‍♀️👷👷‍♂️   <br> 
3.  Build multiple models 👷👷‍♂️👷👷‍♂️  <br>
4.  Train the models 🏋️‍♂️ 🏋️‍♀️ 🏋️‍♂️ 🏋️‍♀️ <br>
5.  Examine models' history (integrate tensorboard) 🔍😍😍😍🔍 <br>
6.  Test the models 😋😋😋<br>
7.  Select the best and promising model and improve it further 🏆🏆🏆<br>
8.  Deploy the winning model 🚀🚀🚀🚀🚀🚀<br>
</h5>
-->

<p>
This is an app built and used to train and test autoencoder models for ieee'22 oceans conference paper (Paper ID 2022169954). 
Only Dense layers were used in the models. I ran out of time to finish the models with CNN layers. 
New models can be trained and saved using this app. However, due to expected resource constraints in the free deployment server, 
the app should be used locally. <span  style="color:red">Even when testing the provided models with the larger datasets can give the server troubles, and the app may crash</span>. 
An image will be added to DockerHub soon to make it easier to run the app locally.
In addition to the hydrophone spectrogram images used in the paper, the models in this repo are trained on WERA radar Range-Cell (RC) images. 
Only a subset, including the final model ( model_build_time 14/08/2022 00:40) presented in the paper, of the models are included.
All data credit to Ocean Networks Canada (ONC).
</p>
"""
st.markdown(preamble, unsafe_allow_html=True)
