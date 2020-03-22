# Ultimate-COVD-19-Detector
The World's best Doctors are currently working on finding and testing the cure to the COVID-19 virus. In the middle of all of this, some countries are unable to find testing kits for the virus, making it really difficult to curb it's spread. In the hopes of helping and contributing an easier detection method, I recently got my hands on a X-Ray Dataset of COVID-19 patients(Huge thanks to <a href=https://github.com/ieee8023>ieee8023</a>). <br>
In this matter, the biggest task for a machine to classify COVID would be to differentiate between a Bacterial Pneumonia and a Viral pneumonia like COVID-19's because both of their symptoms are similar upto a large extent. So here I am showing you my ML model which can differentiate betweeen a normal person, a person with bacterial pneumonia and a person with COVID-19.<br><br>
Testing scores(All metrics are weighted because of the intial class imbalance):
1. <b>Accuracy</b>- 0.8564749883122955
2. <b>Precision</b>- 0.8774582560296846
3. <b>Recall</b>- 0.8311688311688312
4. <b>F1-Score</b> - 0.8289390573366232
5. <b>AUC</b> - 0.9797247840726102 <br><br>
This result has been checked multiple times on unseen images and so far, for all the testing datasets, the model is able to score similarly.<br>
I really hope this would be useful to the Doctors and Medical professionals to help them save lives and time<br>
I would appreciate it if this model is tested on further data because for now the small dataset available might not be a true test for it

# How To Use
1. Install all the requirements by using the following code while in the current directory<br>
```pip install -r requirements.txt```<br>
2. Execute the <a href=https://github.com/DarshanDeshpande/COVID-19-Detector/blob/master/Predict.py>Predict.py</a> file through the terminal or your preferred IDE, give the path to the folder containing the images and done!

# Model Architecture
![Model Architecture](https://github.com/DarshanDeshpande/COVID-19-Detector/blob/master/ModelArchitecture.jpg)

# DATA USED <br>
  1. <a href=https://github.com/ieee8023/covid-chestxray-dataset>COVID-19 Xray images</a>
  2. <a href=https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia>Paul Mooney's Amazing Pneumonia Dataset</a>
  3. Link to all images and TFRecord files used for training, validation and testing can be found here on my drive -> https://drive.google.com/open?id=1rxznEYJljIy8tpKyDitraHRXdlfn1n7u
