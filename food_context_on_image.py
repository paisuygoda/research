import pickle
import glob
import os
import sys
import chainer
from PIL import Image
from chainer.datasets import tuple_dataset
import numpy as np

from gensim.models.keyedvectors import KeyedVectors

genre = [
    "Japanese", "Italian", "French", "Chinese", "Indian", "Korean", "Thai", "Vietnamese", "Singaporean", "Ramen"
]


def img_shrink():
    country = "Vietnam"
    target_size = (224,224)
    if not os.path.exists("C:/Users/Yuji Goda/Documents/gitFiles/TLCrawl/tutorial/spiders/tokyo/{}/mini".format(country)):
        os.mkdir("C:/Users/Yuji Goda/Documents/gitFiles/TLCrawl/tutorial/spiders/tokyo/{}/mini".format(country))
    dir_path = "C:/Users/Yuji Goda/Documents/gitFiles/TLCrawl/tutorial/spiders/tokyo/{}/full/*.jpg".format(country)
    max_size = len(glob.glob(dir_path))
    for i, name in enumerate(glob.glob(dir_path)):
        save_name = str(i)+".jpeg"
        try:
            img = Image.open(name)
        except OSError as e:
            continue
        w, h = img.size
        if w > h :
            blank = Image.new('RGB', (w, w))
        if w <= h :
            blank = Image.new('RGB', (h, h))
        try:
            blank.paste(img, (0, 0) )
        except OSError as e:
            continue
        blank = blank.resize( target_size )
        blank.save("C:/Users/Yuji Goda/Documents/gitFiles/TLCrawl/tutorial/spiders/tokyo/{country}/mini"
                   "/{save_name}".format(save_name=save_name, country=country), "jpeg" )


def make_data():

    path_to_data = "~/home/goda/research/dataset_10000/"
    paths_and_labels = []

    for i in range(10):
        path_each = path_to_data + str(i) + "/"
        paths_and_labels.append(np.asarray([path_each, i]))

    # データを混ぜて、trainとtestがちゃんとまばらになるように。
    all_data = []
    for path_and_label in paths_and_labels:
        path = path_and_label[0]
        label = path_and_label[1]
        image_list = glob.glob(path + "*")
        for img_name in image_list:
            all_data.append([img_name, label])
    all_data = np.random.permutation(all_data)
    print("finished getting path")
    imageData = []
    labelData = []
    max = len(all_data)
    for i, pathAndLabel in enumerate(all_data):
        img = Image.open(pathAndLabel[0])
        #3チャンネルの画像をr,g,bそれぞれの画像に分ける
        r,g,b = img.split()
        rImgData = np.asarray(np.float32(r)/255.0)
        gImgData = np.asarray(np.float32(g)/255.0)
        bImgData = np.asarray(np.float32(b)/255.0)
        imgData = np.asarray([rImgData, gImgData, bImgData])
        imageData.append(imgData)
        labelData.append(np.int32(pathAndLabel[1]))
        sys.stdout.write("\r%d" % i)
        sys.stdout.write(" / %d" % max)
        sys.stdout.flush()

    threshold = np.int32(len(imageData)/8*7)
    train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
    test = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])

    with open('train_image.pickle', mode='wb') as f:
        pickle.dump(train, f)
    with open('test_image.pickle', mode='wb') as f:
        pickle.dump(test, f)

make_data()

