# Image-augmentation-Example

Image-augmentation-example is an example of how to use **albumentations.ai** library to augmentation image to serve image processing in machine learning

## Why need to augmentation images ?
Normally, In object-detection task , you will need a large dataset to have a good result after training process. But in individual cases, you may not have that large of data, so with a quite small amount of data , you can still get a aceptable result if you put that datas in many context that close to reality. Image-augmentation is collection of suck these actions

### Explaination
  > In this selected scenario, I using dataset with Pascal-VOC format in labeling process, and a very good augmentation library named albumentations.ai.
      Following https://albumentations.ai/docs/.
      
  > Our target is create a new image base on existed image but have diferrent context, then copy same .xml file(labeling file) and rename it.
  
  1. Installation
    
   1.1 Albumentations package
    
    pip install -U albumentations
     
   Following https://albumentations.ai/docs/getting_started/installation/
    
   1.2 The ElementTree XML
   
   1.3 OpenCV
