# Image Inpainting using Deep Learning
## Problem overwiev
Image inpainting is a process of restoring parts of image that are broken. The dataset for this problem is musical albums cover arts from [kaggle] (https://www.kaggle.com/datasets/greg115/album-covers-images) containing 80k of 512x512 images (later those will be resized to 256x256). 
Main challenge of this dataset is it's diversity - the cover arts cover many topics, styles and compositions. That's why it was not expected for model to recreate for example band logos, but make them look believable. Every image for this task was corrupted with function that replaces 4 32x32 patches with black pixels. The patches can overlap however which allows for a bit more complex shapes than squares. Since corruptions were done "on the fly" and randomized on each run each image produced a lot of different training images.
# Overfit test
The model was first overfitted on single batch of images (8 images), with broke patches occuring at the same place for each image for every step.
## Looking for architecture
The solution is based on U-Net style architecture, since it allows for retrieving informations from before bottleneck layers. This should help percieve structure and texture. In total 3 architectures of different sizes were tested. They were overfitted on single batch (8 images) and the smallest seemed to be sufficient, although there was slight plateau of loss function at the end so no smaller models were tested.
## Used loss functions
Since using reconstruction loss (L1/L2) loss is insufficient and produces blurry results other loss functions are needed. Alongside reconstruction loss perceptual loss was introduced. It's based on pretrained VGG16 model and it's goal is to reduce bluriness. After this change the blur problem was no more.
## Looking for learning rate
To find learning rate that shall be used for final training a few tests were run. First the best learning rate was found for the overfitted model and 1e-3 seemed to be best. The test was run on exactly the same dataset, with patches in same places to keep consistency. Intutively learning rate for full model should be smaller than found for model above. Second test consisted of training model for 10 epochs with dataset consisting of 1000 images and "normal" random image breaking function. A few candidates seem good, but finally is ended with choosing 3e-4.
## Overfitted model performance

# Full model
Proper model is trained on 20k images with proper breaking function.
## Main problem of trained model
This method produced results that are *sharp* and consistent when it comes to *structure*, but failed on generating realistic textures and colors are slightly off. To address those problems L2 loss was changed to L1 loss and additional Total Variation loss was introduced. It should make the textures smoother and help decrease checkboard artifacts.
# Trying to address those features with architectures change