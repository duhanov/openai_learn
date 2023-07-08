from glob import glob
#from fastai.vision import *
from fastai.vision.all import *
import sys

#data = ImageDataLoaders.from_folder("data/input/porn", 
     #                             train='train', 
    #                              valid='test',
   #                               num_workers=12,
  #                                ds_tfms=get_transforms(), 
 #                                 size=224).normalize(imagenet_stats)
#

     

if not "--path" in sys.argv:
     print("set path")
     exit()

path = sys.argv[sys.argv.index("--path") + 1] if "--path" in sys.argv else ""




#print(get_image_files("input"))
#learn = vision_learner(dls, resnet34, metrics=error_rate).to_fp16()
#learn.fine_tune(1)


label_function = lambda f: str(f).split("/")[-2]


items = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                         get_items=get_image_files, 
                         splitter=RandomSplitter(),
                         get_y=label_function,
                         item_tfms=Resize(224, method=ResizeMethod.Pad))


loader = items.dataloaders(path)


#Проверка закачки
if "--show-batch" in sys.argv:
     loader.show_batch(max_n=9)
     print("loader.show_batch()")


if "--learn" in sys.argv:
     print("Go learn!")     
     learn = cnn_learner(loader, resnet34, metrics=error_rate)
     learn.fine_tune(10)
     lear.save("model")
