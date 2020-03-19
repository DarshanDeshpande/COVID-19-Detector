# Ultimate-COVD-19-Detector
The World's best Doctors are currently working on finding and testing the cure to the COVID-19 virus. In the middle of all of this, some countries are unable to find testing kits for the virus, making it really difficult to curb it's spread. In the hopes of helping and contributing an easier detection method, I recently got my hands on a X-Ray Dataset of COVID-19 patients(Huge thanks to <a href=https://github.com/ieee8023>ieee8023</a>). <br>
In this matter, the biggest task for a machine to classify COVID would be to differentiate between a Bacterial Pneumonia and a Viral pneumonia like COVID-19's because both of their symptoms are similar upto a large extent. So here I am showing you my ML model which can differentiate betweeen a normal person, a person with bacterial pneumonia and a person with COVID-19 with a stunning validation accuracy of over <b>95%</b>. This result has been checked multiple times on unseen images and so far, for all the testing datasets, the model is able to score a<b> perfect score of 100%</b>.<br>
I really hope this would be useful to the Doctors and Medical professionals to help them save lives and time<br>

# How To Use
1. Install all the requirements by using the following code while in the current directory<br>
```pip install -r requirements.txt```<br>
2. Execute the <a href=https://github.com/DarshanDeshpande/COVID-19-Detector/blob/master/Predict.py>Predict.py</a> file through the terminal or your preferred IDE.

# Model Architecture
![alt text](https://github.com/DarshanDeshpande/Ultimate-COVD-19-Detector/blob/master/ModelArchitecture.png)

# DATA USED <br>
  1. <a href=https://github.com/ieee8023/covid-chestxray-dataset>COVID-19 Xray images</a>
  2. <a href=https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia>Paul Mooney's Amazing Pneumonia Dataset</a>
