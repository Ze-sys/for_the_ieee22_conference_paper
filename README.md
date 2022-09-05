<h1  style="color:orange"> <img src="https://theaisummer.com/static/6f5e6b0110a7231a9f70cdf0df9190a0/f4094/topic-autoencoders.png" width=35  height=35 />  Autoencoder  Models Builder
 </h1> 

### A model builder streamlit app for  anomaly detection [![Open app in Streamlit, if deployed](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/ze-sys/autoencoder_model_builder/main/app.py)


<a>
This repo contains a streamlit app that was used to build and train autoencoder models for the ieee'22 oceans conference paper *link to be added*. 
Only Dense layers were used in the models. I ran out of time to finish the models with Convolutional layers. 
The New models can be built and and saved in the *models* directory. However, due to resource constraints, the app need to be run locally. An image will be added to dockerhub soon.
In addition to the hydrophone spectrogram images used in the paper, the models in this repo are trained on WERA radar Range-Cell (RC) images. All data credit to Ocean Networks Canada (ONC).
</a>

<span style="color:black" >
               The model is used to reconstruct the three sets of images (training set- model seen these,
               validation set- not used in back propagation, test set- unseen by the model (all of them are manually 
              checked to be anomalous).
              All images are tested  against the null hypothesis that they represent normal signals. That is, low accuracy is 
              expected for the anomalous set (10% accuracy of predicting anomalous as normal is the same as 90% accuracy of 
              predicting anomalous as anomalous. As such the accuracy and precision of a model  
              in predicting anomalous images as anomalous is reported as fraction of the total number of images 
              sufficiently (based on set loss threshold) reconstructed  subtracted from 1.
</span>

<span style="color:blue" > Credits: [https://www.tensorflow.org/](https://www.tensorflow.org/)
        
</span>



<span style="color:orange" > Reference Links:

 - [https://www.tensorflow.org/tutorials/generative/autoencoder](https://www.tensorflow.org/tutorials/generative/autoencoder)

  - [https://www.jeremyjordan.me/autoencoders/](https://www.jeremyjordan.me/autoencoders/)

   - [https://www.nature.com/articles/s41598-020-80610-9](https://www.nature.com/articles/s41598-020-80610-9)

 </span>