unzip data from TrainINKML_v3.zip

unzip the pretrained.zip folder, which has the following contents 
f_classifier ---> trained Random Forest Classifier for classifying symbols 
f_rel_classifier ---> trained Random Forest Relational classifier

TESTING ONLY PARSING :
For training Relational Classifier on data extracted from inkml files i.e perfect segmentation 
open Relationship_Classifier.py
change the path of the following variable
file_path_till_traininkml ---> to the files where Traininkml files are located
run python Relationship_Classifier.py
After this a parse testing folder will be generated , where the lg files are stored

TESTING ENTIRE SYSTEM:
For Testing the system on classification, segmentation and parsing overall. 
open CLassifier.py
change the path of the following variable
file_path_till_Traininkm ---> to the files where Traininkml files are located
run python Classifier.py
After this a parse testing folder will be generated , where the lg files are stored

