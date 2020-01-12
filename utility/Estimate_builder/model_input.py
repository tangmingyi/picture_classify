import tensorflow as tf
import os
from PIL import Image


def file_based_input_fn_builder(input_file, is_training, drop_remainder, batch):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    input_file = [os.path.join(input_file, name) for name in os.listdir(input_file)]
    name_to_features = {
        "unid": tf.FixedLenFeature([], tf.int64),
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
        "name": tf.FixedLenFeature([], tf.string)
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)  # record是example的序列化，通过这个函数解析为features字典
        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params[batch]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)  # 建立dataset数据的来源
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                # 对每个元素应用map函数进行张量的预处理；dataset可能会将读取的record原始序列张量，传入其中
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def input_fn_builder(config, generator_fn, is_training, drop_remainder, batch):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def _decode_tuple(unid,image,label,name):
        """Decodes a record to a TensorFlow example."""
        res = {}
        res["unid"] = unid
        res["image/encoded"] = image
        res["label"] = label
        res["name"] = name
        return res

    def input_fn(params):
        """The actual input function."""
        batch_size = params[batch]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.Dataset.from_generator(generator=generator_fn,
                                           output_types=(tf.int64, tf.float32, tf.int64, tf.string),
                                           output_shapes=(
                                           tf.TensorShape([]), tf.TensorShape([config["resize"], config["resize"],3]),
                                           tf.TensorShape([]), tf.TensorShape([]))
                                           )
        # d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda unid,image,label,name: _decode_tuple(unid,image,label,name),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d

    return input_fn


def get_generator_fn(config, input_file, data_augment_fn=None):
    def generator_fn():
        for name in os.listdir(input_file):
            data = name[:-4].split("_")
            id = data[-1]
            label = data[0]
            image_path = os.path.join(input_file, name)
            image = Image.open(image_path)
            image = image if data_augment_fn == None else data_augment_fn(image)
            yield id, image.resize((config["resize"], config["resize"]), Image.BILINEAR), label, name
    return generator_fn