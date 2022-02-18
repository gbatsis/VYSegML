# Machine Learning and Multispectral Unmanned Aerial Vehicle Imagery Data for Agriculture.

The emergence of new technological methods in the field of Robotic Systems and Computer Vision has led to the more frequent implementation of Remote Sensing (RS) applications for the automated control of agricultural activities towards Precision Agriculture. In particular, the advent of Unarmed Aerial Vehicles (UAV) offered to the domain experts the convenience to develop RS applications because these systems are a low cost and flexible solution. In this work, a study of Semantic Segmentation for vineyard recognition is presented by combining Multispectral UAV imagery data, Machine Learning (ML) algorithms, Feature Extraction and Feature Selection Methods. The dataset which was used is here https://github.com/Cybonic/DL_vineyard_segmentation_study.git. Concerning Feature Extraction methods, Vegetation Indices and Texture from images were extracted. In order to fit (the Non-Deep) ML methods, it is essential that Feature Extraction Methods should be combined with an efficient sampling method to convert the entire set of images to a pixel-based Dataset. Different ML methods were compared in terms of training and prediction time and their performance and Gaussian Naive Bayes (GNB) was the most efficient method. Despite the fact that GNB was not accurate in the prediction of data containing the entire set of extracted features, accuracy of this method increased after Feature Selection. More precisely, F1 score of GNB combined with Features selected by a tree-based ensemble classifier (Random Forest, AdaBoost) was competitive in comparison with Random Forest, AdaBoost and SVM-RBF and considering its agility during prediction, GNB was finally selected.

Greek Report: https://nbviewer.org/github/gbatsis/VYSegML/blob/main/report.ipynb?flush_cache=true

## Instructions

Setup: git clone https://github.com/gbatsis/VYSegML.git

Create a virtual python environment and run requirements.txt.
Feature Extraction was implemented using a very fast numpy-based method https://github.com/tzm030329/GLCM. This is a submodule. Take its files:
* cd VYSegML
* git submodule init
* git submodule update

### Console Application 

Console Application: You can display results or run everything from the beggining. If you choose to run everything, files in RunTime folder will be replaced.  

run ./src/main.py

WARNING: In your terminal you must be in this path --> path_before/VYSegML

### Dash Application

Dash Application: Contains results and live segmentation of random images from the Test Dataset.

run ./Application/app.py

WARNING: In your terminal you must be in this path --> path_before/VYSegML 


