# Attention MFP-Unet

# Before running the train_segmentation.py

1- Create an empty folder named results_2D.

2- Place your images and masks in the data_2D/images folder.

3- Create a file named image_filemap.csv in the data_2D folder with the following columns:
    
    - images: names of the images
    
    - masks: corresponding masks
    
    - train_test: specify whether each image belongs to the training or testing set. Ensure that the filemap is shuffled.

Once you have completed these steps, you'll be ready to run the train_segmentation.py script and start experimenting with AttMFPUnet.

Happy training!

This code is the implementation of our paper: Automatic fetal biometry prediction using a novel deep convolutional network architecture. If you used it, please cite the paper:

Oghli MG, Shabanzadeh A, Moradi S, Sirjani N, Gerami R, Ghaderi P, Taheri MS, Shiri I, Arabi H, Zaidi H. Automatic fetal biometry prediction using a novel deep convolutional network architecture. Physica Medica. 2021 Aug 1;88:127-37.
    
