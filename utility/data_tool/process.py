import tensorflow as tf
import collections
import json
import os
import pickle
import cv2
import multiprocessing
import random
import numpy as np
from PIL import Image
import Augmentor

conigDir = json.load(open("/home/tmy/programming/picture_classify/config_file/config.json", 'r', encoding='utf-8'))


class Example():
    def __init__(self, picture, label, name):
        self.picture = picture
        self.label = label
        self.name = name


class Feature():
    def __init__(self, unid, picture, label, name):
        self.unid = unid
        self.picture = picture  # (n,c,h,w)
        self.label = label
        self.name = name


class Tool():
    @staticmethod
    def Augmentor(pic_path,num):
        p=Augmentor.Pipeline(pic_path)
        p.random_erasing(probability=0.8,rectangle_area=0.4)
        # p.random_distortion(probability=1,grid_height=4,grid_width=4,magnitude=3)
        # p.shear(probability=1,max_shear_left=15,max_shear_right=15)
        p.sample(num)

    @staticmethod
    def get_raw_data(path):
        with open(path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    @staticmethod
    def get_train_tfrecord():
        out_path = "train"
        num = 1
        for i in range(1, 6):
            path = os.path.join(conigDir["PD"], "data_batch_{}".format(i))
            dataDir = Tool.get_raw_data(path)
            datas = dataDir[b'data']
            labels = dataDir[b'labels']
            filenames = dataDir[b'filenames']
            if not os.path.exists("/".join(["..", "..", conigDir["DP"], out_path])):
                os.mkdir("/".join(["..", "..", conigDir["DP"], out_path]))
            tf_writer = FeatureWriter(
                os.path.join("../../", conigDir["DP"], out_path, "tf_record{}".format(num * 1000)))
            for index, label in enumerate(labels):
                if (index + 1) % 1000 == 0:
                    num += 1
                    tf_writer.close()
                    tf_writer = FeatureWriter(
                        os.path.join("../../", conigDir["DP"], out_path, "tf_record{}".format(num * 1000)))
                tf_writer.process_feature(Feature(index, datas[index], labels[index], filenames[index]))
            tf_writer.close()

    @staticmethod
    def get_train_jpg_tfrecord():
        pic_paths = "/home/tmy/programming/picture_classify/data/raw_jpg_data/train"
        img_paths = []
        for name in os.listdir(pic_paths):
            img_paths.append((os.path.join(pic_paths,name),int(name.split("_")[0])))

        random.shuffle(img_paths)
        num = 1
        tf_writer = FeatureWriter(os.path.join("../../", conigDir["DP"], "train_jpg", "tf_record{}".format(num*1000)))
        for index,data in enumerate(img_paths):
            if (index+1) % 1000 == 0:
                num += 1
                tf_writer.close()
                tf_writer = FeatureWriter(os.path.join("../../", conigDir["DP"], "train_jpg", "tf_record{}".format(num * 1000)))

            pic = tf.gfile.FastGFile(data[0],'rb').read()
            # pic = tf.image.decode_jpeg(pic)
            # pic = cv2.imread(data[0])
            # pic = Image.open(data[0])
            label = data[1]
            tf_writer.process_feature(Feature(index,pic,label,data[0].encode()))
        tf_writer.close()

    @staticmethod
    def get_train_jpg_augmentor_tfrecord():
        pic_paths = "/home/tmy/programming/picture_classify/data/raw_jpg_data/output"
        img_paths = []
        for name in os.listdir(pic_paths):
            img_paths.append((os.path.join(pic_paths,name),int(name.split("_")[2])))

        random.shuffle(img_paths)
        num = 1
        tf_writer = FeatureWriter(os.path.join("../../", conigDir["DP"], "train_augmentor", "tf_record{}".format(num*1000)))
        for index,data in enumerate(img_paths):
            if (index+1) % 1000 == 0:
                num += 1
                tf_writer.close()
                tf_writer = FeatureWriter(os.path.join("../../", conigDir["DP"], "train_augmentor", "tf_record{}".format(num * 1000)))

            pic = tf.gfile.FastGFile(data[0],'rb').read()
            # pic = tf.image.decode_jpeg(pic)
            # pic = cv2.imread(data[0])
            # pic = Image.open(data[0])
            label = data[1]
            tf_writer.process_feature(Feature(index,pic,label,data[0].encode()))
        tf_writer.close()

    @staticmethod
    def get_test_jpg_tfrecord():
        pic_paths = "/home/tmy/programming/picture_classify/data/raw_jpg_data/test"
        img_paths = []
        for name in os.listdir(pic_paths):
            img_paths.append((os.path.join(pic_paths,name),int(name.split("_")[0])))

        random.shuffle(img_paths)
        num = 1
        tf_writer = FeatureWriter(os.path.join("../../", conigDir["DP"], "test_jpg", "tf_record{}".format(num*1000)))
        for index,data in enumerate(img_paths):
            if (index+1) % 1000 == 0:
                num += 1
                tf_writer.close()
                tf_writer = FeatureWriter(os.path.join("../../", conigDir["DP"], "test_jpg", "tf_record{}".format(num * 1000)))
            # pic = cv2.imread(data[0])
            pic = tf.gfile.FastGFile(data[0],'rb').read()
            label = data[1]
            tf_writer.process_feature(Feature(index,pic,label,data[0].encode()))
        tf_writer.close()




    @staticmethod
    def get_test_tfrecord():
        out_path = "test"
        num = 1
        path = os.path.join(conigDir["PD"], "test_batch")
        dataDir = Tool.get_raw_data(path)
        datas = dataDir[b'data']
        labels = dataDir[b'labels']
        filenames = dataDir[b'filenames']
        if not os.path.exists("/".join(["..", "..", conigDir["DP"], out_path])):
            os.mkdir("/".join(["..", "..", conigDir["DP"], out_path]))
        tf_writer = FeatureWriter(os.path.join("../../", conigDir["DP"], out_path, "tf_record{}".format(num * 1000)))
        for index, label in enumerate(labels):
            if (index + 1) % 1000 == 0:
                num += 1
                tf_writer.close()
                tf_writer = FeatureWriter(
                    os.path.join("../../", conigDir["DP"], out_path, "tf_record{}".format(num * 1000)))
            tf_writer.process_feature(Feature(index, datas[index], labels[index], filenames[index]))
        tf_writer.close()

    @staticmethod
    def get_total_example_num():
        total_num = 0
        for i in range(1, 6):
            path = os.path.join(conigDir["PD"], "data_batch_{}".format(i))
            dataDir = Tool.get_raw_data(path)
            labels = dataDir[b'labels']
            total_num += len(labels)
        return total_num

    @staticmethod
    def pickle_dump_pic(pic_path_list: list, output):
        imgs = []
        for pic_path in pic_path_list:
            img = cv2.imread(pic_path)
            img = img.flatten()
            imgs.append(img)
        with open(output, 'wb') as wf:
            pickle.dump(imgs, wf, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def multiprocess_pickle_dump(pic_floder, output, process_num):
        def get_jobs():
            pic_names = os.listdir(pic_floder)
            jobs = {}
            i = 1
            temp = []
            for index, pic_name in enumerate(pic_names):
                temp.append(os.path.join(pic_floder, pic_name))
                if (index + 1) % 1000 == 0:
                    jobs[i] = temp
                    i += 1
                    temp = []
            return jobs

        jobs = get_jobs()
        process_pool = multiprocessing.Pool(processes=process_num)
        for k, job in jobs.items():
            process_pool.apply_async(func=Tool.pickle_dump_pic,
                                     kwds={"pic_path_list": job, "output": output + str(k * 1000)})

    @staticmethod
    def pickle_load_pic(pickle_path):
        with open(pickle_path, 'rb') as rf:
            return pickle.load(rf)

    @staticmethod
    def get_train_jpg_raw_data():
        jpg_path = "/home/tmy/programming/picture_classify/data/raw_jpg_data"
        out_path = "train"
        if not os.path.exists(jpg_path):
            os.mkdir(jpg_path)
        if not os.path.exists(os.path.join(jpg_path,out_path)):
            os.mkdir(os.path.join(jpg_path,out_path))
        for i in range(1, 6):
            path = os.path.join(conigDir["PD"], "data_batch_{}".format(i))
            dataDir = Tool.get_raw_data(path)
            datas = dataDir[b'data']
            labels = dataDir[b'labels']
            # filenames = dataDir[b'filenames']
            for index,label in enumerate(labels):
                cv2.imwrite(os.path.join(jpg_path,out_path,"{}_{}_{}.jpg".format(label,i,index)),datas[index].reshape([3,32,32]).transpose([1,2,0]))

    @staticmethod
    def get_test_jpg_raw_data():
        jpg_path = "/home/tmy/programming/picture_classify/data/raw_jpg_data"
        out_path = "test"
        if not os.path.exists(jpg_path):
            os.mkdir(jpg_path)
        if not os.path.exists(os.path.join(jpg_path,out_path)):
            os.mkdir(os.path.join(jpg_path,out_path))
        path = os.path.join(conigDir["PD"], "test_batch")
        dataDir = Tool.get_raw_data(path)
        datas = dataDir[b'data']
        # filenames = dataDir[b'filenames']
        labels = dataDir[b'labels']
        for index,label in enumerate(labels):
            cv2.imwrite(os.path.join(jpg_path,out_path,"{}_{}.jpg".format(label,index)),datas[index].reshape([3,32,32]).transpose([1,2,0]))







class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename):
        self.filename = filename
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=values))
            return feature

        features = collections.OrderedDict()
        features["unid"] = create_int_feature([feature.unid])
        features["image/encoded"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.picture]))
        features["label"] = create_int_feature([feature.label])
        features["name"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.name]))

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())



    def close(self):
        self._writer.close()


if __name__ == '__main__':
    # TRAIN = "/home/tmy/programming/picture_classify/data/raw_jpg_data/train"
    # Tool.Augmentor(TRAIN,50*10000)
    Tool.get_train_jpg_augmentor_tfrecord()

