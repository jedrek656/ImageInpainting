# Image Inpainting using Deep Learning
## Problem overwiev
Image inpainting is a process of restoring parts of image that are broken. The dataset for this problem is musical albums cover arts from [kaggle](https://www.kaggle.com/datasets/greg115/album-covers-images) containing 80k of 512x512 images (later those will be resized to 256x256). 
Main challenge of this dataset is it's diversity - the cover arts cover many topics, styles and compositions. That's why it was not expected for model to recreate for example band logos, but make them look believable. Every image for this task was corrupted with function that replaces 4 32x32 patches with black pixels. The patches can overlap however which allows for a bit more complex shapes than just squares. Since corruptions were done "on the fly" and randomized on each run each image produced a lot of different training samples.
# Overfit test
The model was first overfitted on single batch of images (8 images), with broke patches occuring at the same place for each image for every step.
## Used loss functions
Since using reconstruction loss (L1/L2) loss is insufficient and produces blurry results other loss functions are needed. Alongside reconstruction loss perceptual loss was introduced. It's based on pretrained VGG16 model and it's goal is to reduce bluriness. After this change the blur problem was no more.
## Looking for architecture
The solution is based on U-Net style architecture, since it allows for retrieving informations from before bottleneck layers. This should help percieve structure and texture. In total 3 architectures of different sizes were tested. They were overfitted on single batch (8 images) and the smallest seemed to be sufficient, although there was slight plateau of loss function at the end so no smaller models were tested.
<img width="607" height="364" alt="ovLoSm" src="https://github.com/user-attachments/assets/8dc4bb0d-18be-4c6f-bc8f-e6fa150b02e0" />
<img width="607" height="365" alt="ovLoMi" src="https://github.com/user-attachments/assets/44fe322f-183f-497e-ac10-58bbf5c0ec98" />
<img width="607" height="368" alt="overfitLossLarge" src="https://github.com/user-attachments/assets/d0ba53d8-81dd-4d1f-af95-d1f281ad97f5" />

Overfitting of in order: small, medium and large network

## Overfitted model performance
### The same dataset as being trained on
<img width="1018" height="672" alt="overfittedSetOg" src="https://github.com/user-attachments/assets/1f951475-f08c-4eba-a251-bd3b09c018c8" />
<img width="1018" height="672" alt="overfittedSetRepaired" src="https://github.com/user-attachments/assets/da246960-f5b0-4f94-b428-9d514269f078" />

In order: original vs fixed images

### Different dataset than being trained on
<img width="1018" height="672" alt="overfittedDiffImagesOg" src="https://github.com/user-attachments/assets/21633f47-c446-47eb-9034-c805d1f2af87" />
<img width="1018" height="672" alt="overfittedDiffImagesFixed" src="https://github.com/user-attachments/assets/ab1db5b3-81ab-4ad6-bc7f-b53be9d4f19f" />

In order: original vs fixed images

The color structure is somehow preserved even for not-seen dataset, but everything else is off.

## Looking for learning rate
To find learning rate that shall be used for final training a few tests were run. First the best learning rate was found for the overfitted model and 1e-3 seemed to be best. The test was run on exactly the same dataset, with patches in same places to keep consistency. Intutively learning rate for full model should be smaller than found for model above. Second test consisted of training model for 10 epochs with dataset consisting of 1000 images and "normal" random image breaking function. A few candidates seem good, but finally is ended with choosing 3e-4.

<img width="1366" height="570" alt="lossTest1000images" src="https://github.com/user-attachments/assets/cd44af9e-fd55-4038-8def-58394c0ac79e" />




# Full model
Proper model is trained on 20k images with proper breaking function.
## Main problem of trained model
This method produced results that are *sharp* and consistent when it comes to *structure*, but failed on generating realistic *textures* and *colors* are slightly off. To address those problems L2 loss was changed to L1 loss and additional Total Variation loss was introduced. It should make the textures smoother and help decrease checkboard artifacts.
## Example of results

<img width="1018" height="672" alt="model1Og" src="https://github.com/user-attachments/assets/d20532ae-0a9c-4763-bb76-72f24185e217" />
<img width="1018" height="672" alt="model1Fixed" src="https://github.com/user-attachments/assets/20e95ec9-1c6d-47a6-9888-63f2d2dc183b" />

In order: original vs fixed images

The local structure (edges and shapes) are preserved really good, but the texture is completely off. The "checkboard" artifacts are also present.


# Trying to address those features with architectures change
## First try: add TV loss, decrease weight for perceptual loss a bit, change L2 loss to L1 loss

<img width="1018" height="672" alt="model3Og" src="https://github.com/user-attachments/assets/476cee2c-3efb-4c10-8c0b-7d46703b62e3" />
<img width="1018" height="672" alt="model3Fixed" src="https://github.com/user-attachments/assets/566e4a17-9701-4aa7-9792-576855d5d7ed" />

## Second try: everything from first try + modify bottleneck for larger respective field and get rid of batch norm

<img width="1018" height="672" alt="model7Og" src="https://github.com/user-attachments/assets/6ee29da9-a36d-42f1-938d-cc32f48a2416" />
<img width="1018" height="672" alt="model7Fixed" src="https://github.com/user-attachments/assets/8842343a-f7ce-448b-a32a-fac36a589ce8" />

In order: original vs fixed images

Losses:
<img width="1223" height="801" alt="lossesFinal" src="https://github.com/user-attachments/assets/03032134-065e-4eae-b168-0f7150622539" />


