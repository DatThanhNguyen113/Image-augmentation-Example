# Image-augmentation-Example

Image-augmentation-example is an example of how to use **albumentations.ai** library to augmentation image to serve image processing in machine learning

## Why need to augmentation images ?
Normally, In object-detection task , you will need a large dataset to have a good result after training process. But in individual cases, you may not have that large of data, so with a quite small amount of data , you can still get a aceptable result if you put that datas in many context that close to reality. Image-augmentation is collection of suck these actions

### Explaination
  > In this selected scenario, I using dataset with Pascal-VOC format in labeling process, and a very good augmentation library named albumentations.ai.
      Following https://albumentations.ai/docs/.
      
  > Our target is create a new image base on existed image but have diferrent context, then copy same .xml file(labeling file) and rename it.
  
  * Installation
    
   1.1 Albumentations package
    
    pip install -U albumentations
     
   Following https://albumentations.ai/docs/getting_started/installation/
    
   1.2 The ElementTree XML
   
   1.3 OpenCV
   
   * Explain line-by-line
   
   1.1 Define image folder path
    
      folder_path = os.path.join('D:\\','OneDrive', 'Tensorflow', 'Image_0306','train3')
      
      dirs = os.listdir( folder_path )
   
   1.2 Define label list
      
      class_labels =  ['Wood']
   
   1.3 Clone exists .xml file to new image
      
      def get_boxes_coordinate(file) :
      coordinate_boxes = []
      tree = ET.parse(file)
      root = tree.getroot()
      for child in root:
          if child.tag == "object":
              box = child[4]
              if box is not None:
                  ordinate = [box[0].text,box[0].text,box[0].text,box[0].text,class_labels[0]]
                  coordinate_boxes.append(ordinate)
      return coordinate_boxes
      
   When augmentation images , .xml file that store boxes coordinate may not change, you just need to copy these file to same groups of image 
   
      def override_xml_content(file,img_file_name,image_file_path) :
        tree = ET.parse(file)
        root = tree.getroot()
        root[1].text = img_file_name
        root[2].text = image_file_path
        tree.write(file)
        
   Then change some value in node to selected images (current image name , current image path) in case you want to do something else with these images
    
