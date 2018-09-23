# coding=utf-8
#
# voc 图像转换,处理
#
#

import os
import xml.dom.minidom as xmldom
import PIL.Image
import numpy
import tensorflow as tf

# 训练集根路径
TRAIN_BASE_DIR = "/Users/tomwang/Downloads/deep_learning/VOC/train/VOCdevkit/VOC2012"
# 标记文件夹名
ANNOTATION_FILE = "Annotations"
# 图片文件夹名
IMAGE_FILE = "JPEGImages"
# 图片目标大小 每次修改后需要重新生成tfrecord
TAGET_SIZE = 500.0
LABLE_SIZE = 20.0


def read_tfrecord(input_dir, batch_size=1):
    """
    读取tfrecord文件
    :param input_dir:
    :return:
    """
    dataset = tf.data.TFRecordDataset(input_dir)
    return dataset.map(_data_format).batch(batch_size).shuffle(50).repeat()


def get_next_batch(dataset):
    """
    获取下一批数据
    1.对图片进行reshape,大小缩放,色相,饱和度处理
    2.对border进行归一化处理(所以要求图片大小缩放后,该border比例不会变)
    3.对classfication进行one-hot处理
    4.过滤 truncated 和 difficult
    :param dataset:
    :return:
    """
    data = dataset.make_one_shot_iterator().get_next()
    # 对图片进行处理
    # 色相,饱和度
    image_raw = tf.decode_raw(data['image_arr'], tf.uint8)

    # annotation
    annotation = tf.decode_raw(data['annotation'], tf.float64)

    # 测试代码
    with tf.Session() as sess:
        name, label, image = sess.run([data['image_name'], annotation, image_raw])
        image_entity = PIL.Image.fromarray(image.reshape((500, 500, 3)), 'RGB')
        image_entity.show()

        label = label.reshape((20, 20, 25))
        print name, image.shape, label.shape


def record_use_tfrecord(output_dir):
    """
    将文件夹下的所有annotation全部以TrRecord方式转换存储
    :return:
    """
    writer = tf.python_io.TFRecordWriter(output_dir)

    # 获取文件夹下所有文件名
    names = os.listdir(os.path.join(TRAIN_BASE_DIR, ANNOTATION_FILE))
    # 遍历所有文件
    for name in names:
        try:
            item = read_file(name)
        except Exception as e:
            print e.message, name
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_name': _tfrecord_byte(bytes(item.file_name)),
                'image_arr': _tfrecord_byte(item.image_arr.tostring()),
                'annotation': _tfrecord_byte(item.annotation.tostring())
            }))

            writer.write(example.SerializeToString())

    writer.close()


def read_file(pic_id, filter_diffcult=True):
    """
    读取文件
    :param pic_id 图片ID
    :return:
    """

    class VOCItem(object):
        pass

    class TargetObj(object):
        pass

    result = VOCItem()

    # 读取annotation文件
    annotation_dir = os.path.join(TRAIN_BASE_DIR, ANNOTATION_FILE, pic_id)
    annotation_obj = xmldom.parse(annotation_dir)
    annotation_element = annotation_obj.documentElement

    # 图片文件名解析
    file_name = _get_content_from_tag(annotation_element, 'filename', force=True)
    result.file_name = file_name

    # 图片大小解析
    size_element = annotation_element.getElementsByTagName('size')[0]
    width = _get_content_from_tag(size_element, 'width', force=True)
    height = _get_content_from_tag(size_element, 'height', force=True)
    depth = _get_content_from_tag(size_element, 'depth', force=True)

    # 图片填充 通道数 * 长 * 宽
    delta_width = TAGET_SIZE - float(width)
    delta_height = TAGET_SIZE - float(height)
    single_channel_size = TAGET_SIZE * TAGET_SIZE
    full_image_dir = os.path.join(TRAIN_BASE_DIR, IMAGE_FILE, file_name)
    image = PIL.Image.open(full_image_dir)

    if delta_height > 0.0 or delta_width > 0.0:
        background = PIL.Image.new('RGB', (int(TAGET_SIZE), int(TAGET_SIZE)), (0, 0, 0))
        background.paste(image, (int(delta_width) / 2, int(delta_height) / 2))
        image = background

    # r, g, b = image.split()
    # r_arr = numpy.array(r).astype(numpy.uint8).reshape(int(single_channel_size))
    # g_arr = numpy.array(g).astype(numpy.uint8).reshape(int(single_channel_size))
    # b_arr = numpy.array(b).astype(numpy.uint8).reshape(int(single_channel_size))
    # image_arr = numpy.concatenate((r_arr, g_arr, b_arr))
    # image_arr = image_arr.reshape((int(depth) * int(single_channel_size), 1))
    result.image_arr = numpy.asarray(image)

    # 图片类目预测信息解析 可能存在多对象
    obj_list = []
    obj_elements = annotation_element.getElementsByTagName('object')
    for obj_element in obj_elements:
        target = TargetObj()
        classfication = _classfication_convert(_get_content_from_tag(obj_element, 'name'))
        truncated = _get_content_from_tag(obj_element, 'truncated')
        difficult = _get_content_from_tag(obj_element, 'difficult')
        target.classfication = classfication

        # 过滤掉不好判断的目标
        if int(difficult) > 0 and filter_diffcult: continue

        # 图片border解析
        bndbox_element = obj_element.getElementsByTagName('bndbox')[0]
        xmin = float(_get_content_from_tag(bndbox_element, 'xmin', force=True))
        xmax = float(_get_content_from_tag(bndbox_element, 'xmax', force=True))
        ymin = float(_get_content_from_tag(bndbox_element, 'ymin', force=True))
        ymax = float(_get_content_from_tag(bndbox_element, 'ymax', force=True))

        # 重新计算边框
        xmin = xmin + delta_width / 2
        xmax = xmax + delta_width / 2
        ymin = ymin + delta_height / 2
        ymax = ymax + delta_height / 2
        target.xmax = xmax / TAGET_SIZE
        target.ymax = ymax / TAGET_SIZE
        target.xmin = xmin / TAGET_SIZE
        target.ymin = ymin / TAGET_SIZE

        # 计算中心点 归一化到 0-1区间内
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        target.centerx = center_x
        target.centery = center_y

        # 计算边框长度
        border_width = abs(xmax - xmin) / TAGET_SIZE
        border_height = abs(ymax - ymin) / TAGET_SIZE
        target.borderx = border_width
        target.bordery = border_height

        obj_list.append(target)

    tuple_size = TAGET_SIZE / LABLE_SIZE

    x_cursor = 0.0
    y_cursor = 0.0

    tensor = []
    while True:
        if tuple_size * y_cursor >= TAGET_SIZE:
            break
        line_tensor = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        while True:
            if tuple_size * x_cursor >= TAGET_SIZE:
                x_cursor = 0.0
                break

            for target in obj_list:
                tuple_list = numpy.zeros((25), dtype=numpy.float32)
                if (target.centerx >= tuple_size * x_cursor) and (target.centerx < tuple_size * (x_cursor + 1)) and (
                        target.centery >= tuple_size * y_cursor) and (target.centery < tuple_size * (y_cursor + 1)):
                    # one-hot编码得到类目向量
                    zero = numpy.zeros((20), dtype=numpy.float32)
                    zero[target.classfication] = 1
                    # 把中心点归一化到该tuple的对应位置
                    # delta_x = target.centerx - tuple_size * x_cursor
                    # delta_y = target.centery - tuple_size * y_cursor
                    # normalize_x = delta_x / tuple_size
                    # normalize_y = delta_y / tuple_size

                    # 将 pc,normalize_x/y, borderx/y,one-hot 拼接成向量
                    now_tuple = [1.0, target.ymin, target.xmin, target.ymax, target.xmax]
                    for z in zero:
                        now_tuple.append(z)
                    line_tensor[int(x_cursor)] = now_tuple
                    break
                else:
                    line_tensor[int(x_cursor)] = tuple_list.tolist()

            x_cursor = x_cursor + 1

        tensor.append(line_tensor)
        y_cursor = y_cursor + 1
    tensor_arr = numpy.array(tensor)
    result.annotation = tensor_arr

    print file_name, tensor_arr.shape

    return result


def _get_content_from_tag(element, tag, eindex=0, nindex=0, force=False, default='0'):
    """
    获取content <tag> content </tag>
    :param element: 节点
    :param tag: 标签
    :return: 内容
    """
    try:
        return element.getElementsByTagName(tag)[eindex].childNodes[nindex].data
    except Exception as e:
        if force:
            raise e
        else:
            print 'occur %s , tag is %s but is not force so can continue it' % (e.message, tag)
            return default


def _data_format(data):
    """
    解析tfrecord中的数据
    :param data:
    :return:
    """
    features = tf.parse_single_example(
        data,
        features={
            'image_name': tf.FixedLenFeature([], dtype=tf.string),
            'image_arr': tf.FixedLenFeature([], dtype=tf.string),
            'annotation': tf.FixedLenFeature([], dtype=tf.string),
        })
    return features


def _classfication_convert(clas):
    """

    :param clas:
    :return:
    """
    if clas == 'aeroplane':
        return 0
    elif clas == 'bicycle':
        return 1
    elif clas == 'bird':
        return 2
    elif clas == 'boat':
        return 3
    elif clas == 'bottle':
        return 4
    elif clas == 'bus':
        return 5
    elif clas == 'car':
        return 6
    elif clas == 'cat':
        return 7
    elif clas == 'chair':
        return 8
    elif clas == 'cow':
        return 9
    elif clas == 'diningtable':
        return 10
    elif clas == 'dog':
        return 11
    elif clas == 'horse':
        return 12
    elif clas == 'motorbike':
        return 13
    elif clas == 'person':
        return 14
    elif clas == 'pottedplant':
        return 15
    elif clas == 'sheep':
        return 16
    elif clas == 'sofa':
        return 17
    elif clas == 'train':
        return 18
    elif clas == 'tvmonitor':
        return 19
    else:
        raise Exception('no mapping type of classfication')


def _tfrecord_byte(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _tfrecord_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _tfrecord_float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


if __name__ == '__main__':
    # read_file('2007_000793.xml')
    record_use_tfrecord(os.path.join(TRAIN_BASE_DIR, 'tfrecord/train.tfrecords'))
    # get_next_batch(read_tfrecord(os.path.join(TRAIN_BASE_DIR, 'tfrecord/train.tfrecords')))
