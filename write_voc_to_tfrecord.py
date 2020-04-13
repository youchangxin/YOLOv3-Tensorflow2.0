import xml.dom.minidom as xdom
import os
import tensorflow as tf
from configuration import cfg


PASCAL_VOC_DIR = "./data/VOC2012/"


class ParsePascalVOC(object):
    def __init__(self):
        self.all_xml_dir = PASCAL_VOC_DIR + "Annotations"
        self.all_image_dir = PASCAL_VOC_DIR + "JPEGImages"

    def __process_coord(self, x_min, y_min, x_max, y_max):
        x_min = int(float(x_min))
        y_min = int(float(y_min))
        x_max = int(float(x_max))
        y_max = int(float(y_max))
        return x_min, y_min, x_max, y_max

    # parse one xml file
    def __parse_xml(self, xml):
        DOMTree = xdom.parse(os.path.join(self.all_xml_dir, xml))
        annotation = DOMTree.documentElement
        image_name = annotation.getElementsByTagName("filename")[0].childNodes[0].data
        size = annotation.getElementsByTagName("size")
        image_height = 0
        image_width = 0
        for s in size:
            image_height = s.getElementsByTagName("height")[0].childNodes[0].data
            image_width = s.getElementsByTagName("width")[0].childNodes[0].data
        obj = annotation.getElementsByTagName("object")
        o_list = []
        for o in obj:
            obj_name = o.getElementsByTagName("name")[0].childNodes[0].data
            bndbox = o.getElementsByTagName("bndbox")[0]

            xmin = bndbox.getElementsByTagName("xmin")[0].childNodes[0].data
            ymin = bndbox.getElementsByTagName("ymin")[0].childNodes[0].data
            xmax = bndbox.getElementsByTagName("xmax")[0].childNodes[0].data
            ymax = bndbox.getElementsByTagName("ymax")[0].childNodes[0].data
            xmin, ymin, xmax, ymax = self.__process_coord(xmin, ymin, xmax, ymax)
            o_list.append(xmin)
            o_list.append(ymin)
            o_list.append(xmax)
            o_list.append(ymax)
            o_list.append(cfg.YOLO.CLASSES[obj_name])

        return image_name, int(image_height), int(image_width), o_list

    # xxx.xml image_height image_width xmin ymin xmax ymax class_type xmin ymin xmax ymax class_type ...
    def __combine_info(self, image_name):
        image_path = self.all_image_dir + "/" + image_name
        return image_path

    def write_data_to_tfrecord(self, tfrecord_path):
        images = []
        height = []
        width = []
        box_list = []
        for item in os.listdir(self.all_xml_dir):
            image_name, image_height, image_width, box = self.__parse_xml(xml=item)

            image_path = self.__combine_info(image_name)
            images.append(image_path)
            height.append(image_height)
            width.append(image_width)
            box_list.append(box)

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            count = 0
            for image, image_height, image_width, box in zip(images, height, width, box_list):
                print("Writing information of picture {}".format(count))
                count += 1
                image = open(image, 'rb').read()  # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    'image_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_height])),
                    'image_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_width])),
                    'boxes': tf.train.Feature(int64_list=tf.train.Int64List(value=box))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())


if __name__ == '__main__':
    ParsePascalVOC().write_data_to_tfrecord(cfg.TRAIN.TFRECORD_DIR)
