unzip data from TrainINKML_v3.zip

unzip pretrained.zip folder, which has the following contents 
f_classifier ---> trained Random Forest Classifier for classifying symbols 
f_rel_classifier ---> trained Random Forest Relational classifier

TESTING ONLY PARSING :
For training Relational Classifier on data extracted from inkml files i.e perfect segmentation 
open Relationship_Classifier.py
change the path of the following variable
file_path_till_traininkml ---> to the files, where Traininkml files are located
run python Relationship_Classifier.py
After this a parse testing folder will be generated , where .lg files are stored

TESTING ENTIRE SYSTEM:
For Testing the system on classification, segmentation and parsing overall. 
open CLassifier.py
change the path of the following variable
file_path_till_traininkml ---> to the files, where Traininkml files are located
run python Classifier.py
After this a parse testing folder will be generated , where .lg files are stored

For Testing, you have to clone following repositories.
git clone http://saskatoon.cs.rit.edu:10001/root/lgeval.git
git clone http://saskatoon.cs.rit.edu:10001/root/crohmelib.git

Once you have setup the above libraries, you first have to use crohme2lg to convert groundtruth MathML(i.e inkml) files to .lg
files (i.e label graph). Now, store this is one folder like groundtruth_out (This will contain .lg file for the corresponding
MATHML(i.e inkml) file)

When you run Classifier.py file, this will generate .lg files for training/testing data, store all the .lg files in a seperate
folder (This will be created and files will be stored in that folder).

To EVALUATE the system 
run evaluate script of lgEval, which takes two directories as input, one containing label graph files to be evaluated, and a
second directory containing (identically named) label graph files providing ground truth.
Metrics, errors, summaries, and visualizations of recognition errors are produced by the script, and stored in a new directory.




