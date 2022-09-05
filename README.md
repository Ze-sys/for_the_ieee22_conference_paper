<h1  style="color:orange"> <img src="https://theaisummer.com/static/6f5e6b0110a7231a9f70cdf0df9190a0/f4094/topic-autoencoders.png" width=35  height=35 />  Autoencoder  Models Builder
 </h1> 
<a> Anomaly detection using basic autoencoders </a>

### [![Open app in Streamlit, if deployed](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ze-sys-for-the-ieee22-conference-paper-home-hr7pb3.streamlitapp.com/)



This repository contains a streamlit app that I built and used to train and test autoencoder models for ieee'22 oceans conference paper (Paper ID 2022169954). 
Only Dense layers were used in the models. I ran out of time to finish the models with CNN layers. 
New models can be trained and and saved using this app. However, due to expected resource constraints in the free deployment server, the app should be used locally. Even when testing the provided models with the **larger datasets can give the server troubles, and the app may crash**. An image will be added to DockerHub soon to make it easier to run the app locally.
In addition to the hydrophone spectrogram images used in the paper, the models in this repo are trained on WERA radar Range-Cell (RC) images. Only a subset, including the final model presented in the paper, of the models are included in this repo.

Details to come soon.


#### Credits: 
 
 - [https://www.tensorflow.org/](https://www.tensorflow.org/)

 -  All data credit to Ocean Networks Canada (ONC)
 

<span style="color:orange" > Reference Links:

 - [https://www.tensorflow.org/tutorials/generative/autoencoder](https://www.tensorflow.org/tutorials/generative/autoencoder)

 - [https://www.jeremyjordan.me/autoencoders/](https://www.jeremyjordan.me/autoencoders/)

  - [https://www.nature.com/articles/s41598-020-80610-9](https://www.nature.com/articles/s41598-020-80610-9)

 </span>