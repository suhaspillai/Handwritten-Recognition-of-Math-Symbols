Step 1: extract the contents from the zip folder 

For just Training the Relational Classifier on data extracted from inkml files i.e perfect segmentation 

do vi Relationship_Classifier.py

change the path of the following 
file_path_till_traininkml ---> to the files where Traininkml files are located

For me the path is 
 file_path_till_traininkml='/home/sbp3624/PatterRecognition/TrainINKML_v3/'

After this a parse testing folder will be generated , where the lg files are stored


For Testing the system on classification, segmentation and parsing overall. 

open CLassifier.py

change the path of the following
file_path_till_Traininkm ---> to the files where Traininkml files are located

For me the path is
 file_path_till_Traininkm='/home/sbp3624/PatterRecognition/TrainINKML_v3/'

After this a parse testing folder will be generated , where the lg files are stored

