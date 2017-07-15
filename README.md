
# Training Faster RCNN on Imagenet
[![Readme Score](http://readme-score-api.herokuapp.com/score.svg?url=andrewliao11/py-faster-rcnn-imagenet)](http://clayallsopp.github.io/readme-score?url=andrewliao11/py-faster-rcnn-imagenet)

If you want to know some basic ideas in faster rcnn, try to check [Video Object Detection using Faster R-CNN](https://andrewliao11.github.io/object/detection/2016/07/23/detection/) out!

Feel free to contact me via email, I'll try to give you a hand if I can, lol.

## preparing data

```
ILSVRC13 
└─── ILSVRC2013_DET_val
    │   *.JPEG (Image files, ex:ILSVRC2013_val_00000565.JPEG)
└─── ILSVRC2013_DET_bbox_val
    |   *.xml (you can find the example from ./misc/ILSVRC2012_val_00018464.xml under this repo)
└─── data
    │   meta_det.mat 
    └─── det_lists
             │  val1.txt, val2.txt
```
meta_det.mat => Load the category inside, like [here](https://github.com/andrewliao11/py-faster-rcnn-imagenet/blob/master/lib/datasets/imagenet.py#L26/)    
Load the meta_det.mat file by 
```
classes = sio.loadmat(os.path.join(self._devkit_path, 'data', 'meta_det.mat'))
```

## Construct IMDB file
There's are several file you need to modify.

#### factory_imagenet.py
This file is in the directory **$FRCNN_ROOT/lib/datasets**($FRCNN_ROOT is the where your faster rcnn locate) and is called by train_net_imagenet.py.  
It is the interface loading the imdb file.   
```
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = '/media/VSlab2/imagenet/ILSVRC13'
    __sets[name] = (lambda split=split, devkit_path=devkit_path:datasets.imagenet.imagenet(split,devkit_path))
```
#### imagenet.py
##### In function __ __init__ __(self, image_set, devkit_path)    
we have to enlarge the number of category from 20+1 into 200+1 categories. Note that in imagenet dataset, the object category is something like "n02691156", instead of "airplane"
```
self._data_path = os.path.join(self._devkit_path, 'ILSVRC2013_DET_' +     self._image_set[:-1])
synsets = sio.loadmat(os.path.join(self._devkit_path, 'data', 'meta_det.mat'))
self._classes = ('__background__',)
self._wnid = (0,)
for i in xrange(200):
    self._classes = self._classes + (synsets['synsets'][0][i][2][0],)
    self._wnid = self._wnid + (synsets['synsets'][0][i][1][0],)
self._wnid_to_ind = dict(zip(self._wnid, xrange(self.num_classes)))
self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
```
self._class denotes the class name   
self._wnid denotes the id of the category

##### In function _load_imagenet_annotation(self, index)
This is because in the pascal voc dataset, all coordinates start from one, so in order to make them start from 0, we need to minus 1. But this is not true for imagenet, so we should not minus 1.   
So we need to modify these lines to:
```
for ix, obj in enumerate(objs):
    x1 = float(get_data_from_tag(obj, 'xmin'))
    y1 = float(get_data_from_tag(obj, 'ymin'))
    x2 = float(get_data_from_tag(obj, 'xmax'))
    y2 = float(get_data_from_tag(obj, 'ymax'))
    cls = self._wnid_to_ind[str(get_data_from_tag(obj, "name")).lower().strip()]
```
Noted that in faster rcnnn, we don't need to run the selective-search, which is the main difference from fast rcnn.
## Modify the prototxt
Under the directory **$FRCNN_ROOT/**   
#### train.prototxt   
Change the number of classes into 200+1
```
param_str: "'num_classes': 201"
```
In layer "bbox_pred", change the number of output into (200+1)*4
```
num_output: 804
```
You can modify the **test.prototxt** in the same way. 

## [Last step] Modify the shell script
Under the dircetory **$FRCNN_ROOT/experiments/scripts**   
#### faster_rcnn_end2end_imagenet.sh
You can specify which dataset to train/test on and your what pre-trainded model is
```
ITERS=100000
DATASET_TRAIN=imagenet_val1
DATASET_TEST=imagenet_val2
NET_INIT=data/imagenet_models/${NET}.v2.caffemodel
```
## Start to Train Faster RCNN On Imagenet!
Run the **$FRCNN/experiments/scripts/faster_rcnn_end2end_imagenet.sh**.   
The use of .sh file is just the same as the original [faster rcnn ](https://github.com/rbgirshick/py-faster-rcnn)

## Experiment
This is the mean/median AP of different iterations.The highest mean AP falls in 90000 iterations.   
![](https://github.com/andrewliao11/py-faster-rcnn/blob/master/asset/mAP_imagenet.png?raw=true)   

The original Faster R-CNN states that they can achieve **59.9% mAP** on PASCAL VOC 2007, which only contains 20 categories. The result of mine is relatively low compared to the original work. However, this is the trade-off since we increase the diversity of the object categories. My network can achieve **33.1% mAP**.  

The low accuracy is due to:   
- Smaller dataset( ImageNet validation1 )
- Diverse object category   

So here I present the result of the overlapped category. My model achieves **48.7% mAP** from the object category that appears in PASCAL VOC 2007 (12 categories), which is much higher than that of 200 categories.
![](https://github.com/andrewliao11/py-faster-rcnn/blob/master/asset/mAP_overlap.png?raw=true)   

And I also present the mAP for each category in ImageNet   
![](https://github.com/andrewliao11/py-faster-rcnn/blob/master/asset/mAP_200.png?raw=true)

## Demo
Just run the **demo.py** to visualize pictures! 
![demo_02](https://github.com/andrewliao11/py-faster-rcnn/blob/master/tools/output_demo_02.jpg?raw=true)

### faster rcnn with tracker on videos
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/wY7LADoEuFs/0.jpg)](http://www.youtube.com/watch?v=wY7LADoEuFs)

Original video "https://www.jukinmedia.com/videos/view/5655"
## Reference
[How to train fast rcnn on imagenet](http://sunshineatnoon.github.io/Train-fast-rcnn-model-on-imagenet-without-matlab/)
