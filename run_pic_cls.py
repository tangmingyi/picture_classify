# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from lib.se_resnet_xt.moding import SE_ResNet_Xt
from lib.se_resnet_xt import optimization, myhook
from utility.data_tool import process
import tensorflow as tf
from tensorflow.python import debug as tfdbg

configDir = json.load(open("config_file/config.json", "r", encoding="utf-8"))


def file_based_input_fn_builder(input_file, is_training, drop_remainder, batch):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unid": tf.FixedLenFeature([], tf.int64),
        "picture": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
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

def input_fn_builder(input_file, is_training, drop_remainder, batch):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unid": tf.FixedLenFeature([], tf.int64),
        "picture": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
    }


    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params[batch]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d

    return input_fn

def create_model(input, layers, class_dim):
    """Creates a classification model."""
    model = SE_ResNet_Xt(input, layers, class_dim)
    cls_layers = model.get_cls_layer()
    return cls_layers


def model_fn_builder(lays, class_dim, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu=False
                     ):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        unids = features["unid"]
        picture = features["picture"]
        picture = tf.decode_raw(picture, tf.uint8)
        picture = tf.reshape(picture, [-1, 3, 32, 32])
        picture = tf.transpose(picture, [0, 2, 3, 1])
        label = features["label"]

        cls = create_model(picture, lays, class_dim)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None

        def compute_loss(logits, positions):
            one_hot_positions = tf.one_hot(
                positions, depth=class_dim, dtype=tf.float32)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            loss = tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
            return loss

        per_example_loss = compute_loss(cls, label)

        if mode == tf.estimator.ModeKeys.TRAIN:
            total_loss = -tf.reduce_mean(per_example_loss, axis=-1)

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, -1)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                export_outputs=None,
                training_chief_hooks=None,
                # training_hooks=[
                # evalute_hook(handle=handle,feed_handle=test_handle, run_op=total_loss, evl_step=10),
                # train_hook(handle,train_handle)
                # ],
                scaffold=scaffold_fn
            )
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label, predictions=predictions, weights=None)
                loss = tf.metrics.mean(values=per_example_loss, weights=None)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label, cls])
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=-tf.reduce_sum(per_example_loss, axis=-1),
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "unique_ids": unids,
                "predict": tf.arg_max(cls, -1),
                "label": label,
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions, scaffold=scaffold_fn)
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn





def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tf_run_config = tf.estimator.RunConfig(
        model_dir=configDir["model_dir"],
        tf_random_seed=None,
        save_summary_steps=configDir["save_summary_steps"],
        save_checkpoints_steps=configDir["save_checkpoints_steps"],
        session_config=None,
        keep_checkpoint_max=configDir["keep_checkpoint_max"],
        log_step_count_steps=configDir["print_loss_steps"],
        train_distribute=None,
        device_fn=None

    )

    num_train_steps = None
    num_warmup_steps = None
    if configDir["do_train"] == 1:
        train_examples = process.Tool.get_total_example_num()
        num_train_steps = int(
            len(train_examples) / configDir["train_batch_size"] * configDir["num_train_epochs"])
        num_warmup_steps = int(num_train_steps * configDir["warmup_proportion"])

    model_fn = model_fn_builder(lays=configDir["SeResnetXt_layers"], class_dim=configDir["class_dim"],
                                learning_rate=configDir["learning_rate"], num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf_run_config,
        params={"train_batch_size": configDir["train_batch_size"],
                "eval_batch_size": configDir["val_batch_size"],
                "predict_batch_size": configDir["test_batch_size"]},  # params可以传给mofel_fn和input_fn
        warm_start_from=None,
    )

    # 是否生成推断模型。
    if configDir["save_predict_model_for_tfServing"] == 1:
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn({
            "unid": tf.FixedLenFeature([], tf.int64),
            "picture": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        })
        estimator.export_savedmodel(configDir["TFServing_model_path"], serving_input_receiver_fn,
                                    strip_default_attrs=True)
        return 0

    if configDir["do_train"] == 1:
        input_files = os.listdir(os.path.join(configDir["DP"], "train"))
        train_input_fn = file_based_input_fn_builder(input_file=input_files, is_training=True, drop_remainder=True,
                                                     batch="train_batch_size")
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps,
                        hooks=[tfdbg.TensorBoardDebugHook(grpc_debug_server_addresses="localhost:11111"),
                               myhook.timeline_hook(with_one_timeline=True)])

    if configDir["do_test"] == 1:

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Batch size = %d", configDir["test_batch_size"])

        input_files = os.listdir(os.path.join(configDir["DP"], "test"))
        predict_input_fn = file_based_input_fn_builder(input_file=input_files, is_training=False, drop_remainder=False,
                                                       batch="predict_batch_size")


        wf = open(configDir["test_res_output"],"w",encoding="utf-8")
        for mm,result in enumerate(estimator.predict(
                predict_input_fn, yield_single_examples=True
                # ,hooks=[tf_debug.LocalCLIDebugHook(ui_type="readline")]
        )):
            # if len(all_results) % 1000 == 0:
            tf.logging.info("Processing example: %d" % (mm))
            # wf.write(json.dumps(result)+"/n")
            example_id = result["unique_ids"].decode()
            predict = result["predict"]
            label = result["label"]
            wf.write("\t".join([example_id,predict,label])+"\n")
        wf.close()


if __name__ == "__main__":
    tf.app.run()
