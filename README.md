# Classification of contacts in protein structures

Angela Kralevska (2072021), Elena Stefanovska (2085310)
---

This is a repo for the final project for the Structural Bioinformatics course.

The **Predictor software** is contained in the **contacts_classifier.py** script. Directions for execution: 
- It should be called with **four arguments** that represent the location of the input **pdb file**, **configuration file**, **output directory**, and **model**.
- In this repo are given examples of pdb file (2ghv.cif), configuration file (configuration.json), and the **final model is also attached (final_model.f5)**.
- In order to be able to execute the script please make sure to specify the **exact location of the dssp_file**, **rama_file** and **atchley_file** 
in the **configuration.json** script.
- Make sure the python libraries: BioPython, Tensorflow, Numpy and Pandas are installed before running the script. 

Output from the **contacts_classifier.py** script:
- Example on how the output from our software looks like can be observed at the **2ghv.tsv file**.
- The last three columns represent the predictions from the model.
- The column named **model_predictions** contains the predictions made from the model in a list of 1s and 0s (where 1 means that the class is predicted, and 0 means that the class is not predicted), such that the classes are ordered in this way -> ["HBOND", "IONIC", "PICATION", "PIPISTACK", "SSBOND", "VDW"].
- The column named **prediction_scores** represents the probabilities outputed from the model, i.e. the confidence scores for its predictions, and they are ordered in the same order as in the **model_predictions** column -> ["HBOND", "IONIC", "PICATION", "PIPISTACK", "SSBOND", "VDW"].
- The last column **predicted_classes** contains list of the names of the predicted classes only.

Instructions for reproducing the model:
- The model can be reproduced by executing the **Final_Model_Training.ipynb** notebook on Google Colab. Make sure to change the **path** variable before executing.
- The dataset used for training can be obtained from this link (https://drive.google.com/file/d/1vuIvAs_rdM-hfeePw7gPhe2TWJKbXf9k/view?usp=sharing), it couldn't be attached on github because it is larger than 100MB. The dataset should be placed in the folder specified at the **path** variable.
- The model will be saved as **final_model.h5** at the same folder specified at **path**.

Experiments with OneVSRest approach for classification and obtaining feature importance by training Random Forest model are available in the **OneVSRest_and_Feature_importance_experiments.ipynb** notebook. 

All experiments and evaluation with the DNN model for multi-label classification are available in the **DNN_experiments.ipynb** notebook. In addition, this notebook contains the final code and plots regarding the EDA (Exploratory Data Analysis) steps as well as the code for calculating the new features and generating the final clean dataset with all features together. Moreover, the code and experiments with MLSMOTE oversampling are included.

