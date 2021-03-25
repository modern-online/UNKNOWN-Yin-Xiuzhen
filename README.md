# UNKNOWN-Yin-Xiuzhen
Parts of code developed/adapted for an interactive installation by Yin Xiuzhen. The installation is comprised out of several parts, and within the current setting all of them are interdependent and required to run. Other parts of the code can be found in [Fito Segrera's repository](https://github.com/fitosegrera/unknown), whereas assets developed in Unity were made by Miller Klistner and have not been published so far. 

## 1. Emotion Recognition

The emotion-recognition script in this repository includes adapted code originally cloned from [Multimodal-Emotion-Recognition](https://github.com/maelfabien/Multimodal-Emotion-Recognition) by maelfabien. It has been adapted for OSC communication as well as made headless due to hardware resource constraints. The models trained in the original repository worked for us hence were not further fine-tuned. To run this script please download the models from maelfabien's repository. 

### Installation

It is recommended to set up a Python virtual environment then install from the supplied requirements.txt file. 
##### pip install -r requirements.txt

#### NOTES: 

If installation of dlib fails on Linux, you need to first install cmake:
##### sudo apt-get install build-essential cmake

Make sure python development dependencies and pip package manager are installed too:
##### sudo apt-get install python3-dev python3-pip

Finally, if required Tensorflow version is not found, in theory any recent Tensorflow should work, so just put a different version in the requirements file. 
