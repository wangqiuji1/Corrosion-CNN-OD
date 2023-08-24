# Corrosion-CNN-OD
This is a code about the corrosion area location and corrosion degree detection of carbon steel materials.
# Requirements environment
For the better use of this code, it is recommended torch>=1.7.1.
# Dataset preparation
1. This code is trained using the VOC format.
   The tag files are placed in annotations under the VOC2007 folder under the VOCdevkit folder.
   The image files are placed in JPEGImages under the VOC2007 folder under the VOCdevkit folder.
2. After completing step a, run voc_annotation.py to get 2007_train.txt and 2007_val.txt for training.
3. Create a self_classes.txt to write the categories you need to distinguish.
# Training
Run train.py to start training. 
In train.py, classes_path is used to point to the self_classes.txt corresponding to the detection class.
# Testing
1. Modify model_path and classes_path in the yolo.py file. 
   The model_path corresponds to the weight file, and the classes_path is the class to which model_path 
   corresponds.
2. Run predict.py.

