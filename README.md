# Attention MFP-Unet

This code is the implementation of our paper: Automatic fetal biometry prediction using a novel deep convolutional network architecture.

# Before running the train_segmentation.py

1- Create an empty folder named results_2D.

2- Place your images and masks in the data_2D/images folder.

3- Create a file named image_filemap.csv in the data_2D folder with the following columns:
    
    - images: names of the images
    
    - masks: corresponding masks
    
    - train_test: specify whether each image belongs to the training or testing set. Ensure that the filemap is shuffled.

Once you have completed these steps, you'll be ready to run the train_segmentation.py script and start experimenting with AttMFPUnet.

Happy training!
    
