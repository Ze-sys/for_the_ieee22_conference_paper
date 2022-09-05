
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
<a>
Only Dense layers were used in the models. I ran out of time to finish the models with Convolutional layers. 
New models can be built and and saved in the *models* directory. However, due to resource constraints, the app need to be run locally. An image will be added to dockerhub soon.
In addition to the hydrophone spectrogram images used in the paper, the models in this repo are trained on WERA radar Range-Cell (RC) images. All data credit to Ocean Networks Canada (ONC).
</a>
"""
st.markdown(preamble, unsafe_allow_html=True)
