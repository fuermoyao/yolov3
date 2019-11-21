import xml.etree.ElementTree as ET
import os
cwd = os.getcwd()

fclass = open('./classes.names', "r")
names = fclass.read().split("\n")[:-1]

def convert_annotation(vocdir,year,modeltxt='train.txt'):
    fimgs = open(modeltxt,'w')
    images = os.path.join(cwd,vocdir+'/VOCdevkit/'+year+'/JPEGImages')
    annotations = os.path.join(cwd,vocdir+'/VOCdevkit/'+year+'/Annotations')
    labels = os.listdir(annotations)
    for label in labels:
        tree = ET.parse(annotations+'/'+label)
        root = tree.getroot()
        f = open(cwd+'/'+vocdir+'/VOCdevkit/'+year+'/Annotations/'+label.split('.')[0]+'.txt','w')
        filename = root.find('filename').text
        fimgs.write(images+'/'+filename+'\n')
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        for obj in root.iter('object'):
            name = obj.find('name').text
            if name not in names:
                continue
            name_id = names.index(name)
            bndbox = obj.find('bndbox')          
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text) 
            x = (xmin+xmax)/2.0 - 1 
            y = (ymin+ymax)/2.0 - 1 
            w = (xmax-xmin)
            h = (ymax-ymin)       
            f.write(str(name_id)+' '+str(x/width)+' '+str(y/height)+' '
                    +str(w/width)+' '+str(h/height)+'\n')
        f.close()
    fimgs.close()

convert_annotation('VOCtrainval_11-May-2012','VOC2012')
convert_annotation('VOC2012test','VOC2012','valid.txt')
fclass.close()
print('done')




# from xml.dom.minidom import parse
# import xml.dom.minidom
# import os
# cwd = os.getcwd()

# fclass = open('./classes.names', "r")
# names = fclass.read().split("\n")[:-1]
# fimgs = open('./train.txt','w')
# images = os.path.join(cwd,'VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages')
# annotations = os.path.join(cwd,'VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations')
# labels = os.listdir(annotations)
# for label in labels:
#     DOMTree = xml.dom.minidom.parse(annotations+'/'+label)
#     collection = DOMTree.documentElement
#     f = open(cwd+'/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/'+label.split('.')[0]+'.txt','w')
#     filename = collection.getElementsByTagName('filename')[0].childNodes[0].data
#     fimgs.write(images+'/'+filename+'\n')
#     size = collection.getElementsByTagName('size')
#     width = float(size[0].getElementsByTagName('width')[0].childNodes[0].data)
#     height = float(size[0].getElementsByTagName('height')[0].childNodes[0].data)
#     objects = collection.getElementsByTagName('object')
#     for obj in objects:
#         name = obj.getElementsByTagName('name')[0].childNodes[0].data
#         name_id = names.index(name)
#         bndbox = obj.getElementsByTagName('bndbox')
#         for i in bndbox:
#             xmin = float(i.getElementsByTagName('xmin')[0].childNodes[0].data)
#             ymin = float(i.getElementsByTagName('ymin')[0].childNodes[0].data)
#             xmax = float(i.getElementsByTagName('xmax')[0].childNodes[0].data)
#             ymax = float(i.getElementsByTagName('ymax')[0].childNodes[0].data) 
#             x = (xmin+xmax)/2
#             y = (ymin+ymax)/2 
#             w = (xmax-xmin)
#             h = (ymax-ymin)       
#             f.write(str(name_id)+' '+str(x/width)+' '+str(y/height)+' '
#                     +str(w/width)+' '+str(h/height)+'\n')
#     f.close()
# fclass.close()
# fimgs.close()
# print('done')
