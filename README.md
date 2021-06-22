# Image-augmentation-Example

Image-augmentation-example is an example of how to use **albumentations.ai** library to augmentation image to serve image processing in machine learning

## Why need to augmentation images ?
Normally, In object-detection task , you will need a large dataset to have a good result after training process. But in individual cases, you may not have that large of data, so with a quite small amount of data , you can still get a aceptable result if you put that datas in many context that close to reality. Image-augmentation is collection of suck these actions

### Explaination
  > In this selected scenario, I using dataset with Pascal-VOC format in labeling process, and a very good augmentation library named albumentations.ai.
      Following https://albumentations.ai/docs/.
      
  > Our target is create a new image base on existed image but have diferrent context, then copy same .xml file(labeling file) and rename it.
  
  * **Installation**
    
   1.1 Albumentations package
    
    pip install -U albumentations
     
   Following https://albumentations.ai/docs/getting_started/installation/
    
   1.2 The ElementTree XML
   
   1.3 OpenCV
   
   * **Explain line-by-line**
   
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
   
   1.4 Define augmentation method
   
   There are so many method in **albumentations.ai** (about 70), each of method has different parameter , whenever you adjust a little bit of them , it will show a different result in image. Just pick up some method that fit your real situation, but be careful, if you abuse many method rabidly , its cause bad prediction result after training (e.g : you pick a method that make random light, but which a very high lighting rate that cannot appear in real-life, and these image can disturb learning process)
   
      random_weather = {
      "RandomRain" : lambda : A.RandomRain(brightness_coefficient = 0.9,drop_width=1,blur_value =5 , p = 1 ),
      "RandomSunFlare" :  lambda : A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5),angle_lower = 0.5,p=1),
      "RandomShadow" :  lambda : A.RandomShadow(num_shadows_lower=1,num_shadows_upper=1,shadow_dimension=5,shadow_roi=(0, 0.5, 1, 1),p=1),
      "Light" : [lambda :A.RandomBrightnessContrast(p=1),lambda :A.RandomGamma(p=1),lambda :A.CLAHE(p=1)]
      }
      
   For easy to usage, we methods that we want to use in a list with selected key, even if you want a group of method in 1 key, just need to put them in a array(e.g : "Light"). These method params can find in **albumentations.ai** (https://albumentations.ai/docs/examples/). Just spend a few minute to read the docs and get what you want
   
   1.5 Augmentation process
    
      for item in dirs:
        file_path = folder_path+"\\"+item
        if os.path.isfile(file_path):
            if file_path.endswith(('.png', '.jpg', '.jpeg')) :
                image = cv2.imread(file_path)
                bboxes = get_boxes_coordinate(file_path.replace('jpg','xml'))
                for attr, value in random_weather.items() : 
                    compose_inside = []
                    if type(value) == list : 
                        for val in value :
                            compose_inside.append(val())
                    else :
                        compose_inside.append(value())
                    transform = A.Compose(compose_inside, bbox_params=A.BboxParams(format='pascal_voc'))
                    transformed = transform(image=image, bboxes=bboxes)
                    new_img_aumentation_name = item.replace('.jpg','')+ "_" + attr + '.jpg'
                    augmentation_name = folder_path + "\\" + new_img_aumentation_name
                    print("save file to : {0}".format(augmentation_name))
                    xml_item = item.replace('jpg','xml')
                    print(xml_item)
                    new_xml_name = xml_item.replace('.xml','')+ "_"+attr + '.xml'
                    new_xml_path = folder_path+"\\" + new_xml_name
                    shutil.copyfile(folder_path+"\\"+xml_item, new_xml_path)
                    override_xml_content(new_xml_path,new_img_aumentation_name,augmentation_name)
                    cv2.imwrite(augmentation_name,transformed['image'])
    
   First, loop through dataset folder and pick files with [jpg] format (or png, jpeg ) and using OpenCV to read them. 
   
   Then **using get_boxes_coordinate()** function created above to get .xml boxes location, we will use that to copy new augmentation-image and pass to augmentation-function
   
   Then, loop through list of augmentation method created above and add them to **A.Compose()** (a **albumentations.ai** method) to create **Transformation** object
   
   Then, pass our's origin image and it's boxes coordinate to created **Transformation**. It will return an object that hold new image (you can log this object to see what it hold)
   
   Then, put the name to new image, I use origin name and append selected augmentation method but you can also use something else, do the same thing with new .xml file we copied above and using **override_xml_content()** function to override new infomation if you want.
   
   Finally, using OpenCV to write that file to current folder again
   
   The reson I  use OpenCV to read and write image because it already installed when using Tensorflow or Pytorch to training object, but you can use different thing, as long as it can read and write image. And that it, now you just access you folder and should see  new images and new .xml files 
   
   
Happy Coding ! :)
