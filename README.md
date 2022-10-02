# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
Malaria is a deadly, infectious mosquito-borne disease caused by Plasmodium parasites. These parasites are transmitted by the bites of infected female Anopheles mosquitoes. Here we use a deep learning technique called CNN to automatically extract the feautures from the cell image and automatically learn useful knowledge that is used to classify the cells as parasatized or uninfected. The dataset is created by Lister Hill National Center for Biomedical Communications (LHNCBC), part of National Library of Medicine (NLM).They have carefully collected and annotated this dataset of healthy and infected blood smear images.
![image](https://user-images.githubusercontent.com/63336975/193441790-66f5172c-8fec-46d6-be8f-2e34e7c97516.png)


## Neural Network Model

![image](https://user-images.githubusercontent.com/63336975/193441816-8e4128b1-73c4-4342-8bf8-39eb1b56bd29.png)

## DESIGN STEPS

### STEP 1:
Download and load the dataset to colab. After that mount the drive in your colab workspace to access the dataset.

### STEP 2:
Use ImageDataGenerator to augment the data and flow the data directly from the dataset directory to the model.

### STEP 3:
Split the data into train and test.

### STEP 4:
Build the convolutional neural network

### STEP 5:
Train the model with training data

### STEP 6:
Evaluate the model with the testing data

### STEP 7:
Plot the performance plot


## PROGRAM

https://github.com/kishore-1812/malaria-cell-recognition-063/blob/main/Copy_of_Ex04.ipynb

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/63336975/193441909-756f2d5d-d6b3-46e3-9891-fe2caaed798f.png)


### Classification Report

![image](https://user-images.githubusercontent.com/63336975/193441925-8b6b753d-3cac-490c-91e3-c235f24c85d1.png)

### Confusion Matrix

![image](https://user-images.githubusercontent.com/63336975/193441958-4298b882-98ec-4657-b07f-3bbb712e2168.png)

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/63336975/193442143-c1294222-819e-44a2-b9ed-03f7710c18f3.png)
![image](https://user-images.githubusercontent.com/63336975/193442157-e7e8bfd3-02e2-440e-96da-35c90a41e439.png)


## RESULT
Successfully developed a convolutional deep neural network for Malaria Infected Cell Recognition.

