# COVID-19-Detector
The World's best Doctors are currently working on finding and testing the cure to the COVID-19 virus. In the middle of all of this, some countries are unable to find testing kits for the virus, making it really difficult to curb it's spread. In the hopes of helping and contributing an easier detection method, I recently got my hands on a X-Ray Dataset of COVID-19 patients(Huge thanks to <a href=https://github.com/ieee8023>ieee8023</a>). <br>
In this matter, the biggest task for a machine to classify COVID would be to differentiate between a Bacterial Pneumonia and a Viral pneumonia like COVID-19's because both of their symptoms are similar upto a large extent. So here I am showing you my ML model which can differentiate betweeen a normal person, a person with bacterial pneumonia and a person with COVID-19.<br>
There is also a model for binary classification for COVID-19 Diagnostics made for research purposes

![Depthwise-Convolution](https://github.com/DarshanDeshpande/COVID-19-Detector/blob/master/images/ModelExpanded.png) <br>
<img src=https://github.com/DarshanDeshpande/COVID-19-Detector/blob/master/images/Visualisation.png width=450 height=450>  <img src=https://github.com/DarshanDeshpande/COVID-19-Detector/blob/master/images/GradientVisualisations/COVID-19-6.jpeg width=250 height=450>

# Testing scores(All metrics are weighted because of the intial class imbalance):
<b>Binary Classification</b><br>
   1.<b>Accuracy</b>- 0.9545 <br>
   2. <b>Precision</b>- 0.9142 <br>
   3. <b>Recall</b>- 0.9696 <br>
   4. <b>F1-Score</b> - 0.9411 <br>
   5. <b>AUC</b> - 0.9959 <br><br>
   <b>Multi-Label Classification</b><br>
   1.<b>Accuracy</b>- 0.8564 <br>
   2. <b>Precision</b>- 0.8774<br>
   3. <b>Recall</b>- 0.8311 <br>
   4. <b>F1-Score</b> - 0.8289 <br>
   5. <b>AUC</b> - 0.9797 <br><br>

This result has been checked multiple times on unseen images and so far, for all the testing datasets, the model is able to score similarly.<br>
I really hope this would be useful to the Doctors and Medical professionals to help them save lives and time<br>
I would appreciate it if this model is tested on further data because for now the small dataset available might not be a true test for it

# How To Use
1. Install all the requirements by using the following code while in the current directory<br>
```pip install -r requirements.txt```<br>
2. Execute the <a href=https://github.com/DarshanDeshpande/COVID-19-Detector/blob/master/Predict.py>Predict.py</a> file through the terminal or your preferred IDE, give your choice of Classification and path of the directory. Done!<br>
```
1. 'Binary' for Binary Classification
2. 'Multi-Label' for Multi-Class Classification 
```
3. After the individual results are displayed, you can choose whether to visualise the gradients of the model. If you do then a window will pop up with the image and it's attention gradients. You can move on to the next image by pressing the ESC key.

# Model Architecture:
The model uses Depthwise Convolutions extensively along with Convolutions in a unit. The features are first extracted using a 2D Convolution layer which are then passed on to the Depthwise Layer where the model learns about features like hazy areas need more attention and focuses more attention on the centre and edges of the Chest (near the cardiophrenic and costophrenic angle areas)as a common trend used by doctors too. These features are now convolved again through another set of filters. This unit is coupled with similar units and repeated thrice. This configuration has proved to be extremely efficient and has yet turned out to be the best Binary model of mine. <br>
<b> NOTE: This has only been tested on unknown sample images received from <a href=https://github.com/ieee8023/covid-chestxray-dataset>@ieee8023's</a> and <a href=https://twitter.com/ChestImaging/status/1243928581983670272>this</a> twitter post's images. Further testing is essential before concrete claims <b>
   
Full Architecture is attached below: <br>
![Model Architecture](https://github.com/DarshanDeshpande/COVID-19-Detector/blob/master/images/ModelArchitecture.png)


# DATA USED <br>
  1. <a href=https://github.com/ieee8023/covid-chestxray-dataset>COVID-19 Xray images</a>
  2. <a href=https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia>Paul Mooney's Amazing Pneumonia Dataset</a>
  3. <a href=https://twitter.com/ChestImaging/status/1243928581983670272> Twitter dataset for Spanish patients</a>
  3. Link to all images and TFRecord files used for training, validation and testing can be found here on my drive -> https://drive.google.com/open?id=1rxznEYJljIy8tpKyDitraHRXdlfn1n7u
  
# CREDITS <br>
1. Adrian Rosebrock for his article for <a href= https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/>grad-cam</a> 
