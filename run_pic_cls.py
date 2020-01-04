"""
todo: add mixup blinear dcl constract
doing: 卷积特征图可视化
"""
import json
import os
from lib.ResNet.model import Model
from lib.ResNet import optimization,myhook,Loss
from lib.ResNet.myhook import BeholderHook
# from utility.data_tool import process
import tensorflow as tf
import numpy as np
# from tensorboard.plugins.beholder import BeholderHook
import cv2
import matplotlib.pyplot as plt


from tensorflow.python import debug as tfdbg

configDir = json.load(open("config_file/config.json", "r", encoding="utf-8"))
model_config = json.load(open(configDir["mode_config"], 'r', encoding='utf-8'))

if configDir["do_save_conv_image"]:
    configDir["test_batch_size"] = 1

if not configDir["do_test"]:
    configDir["do_save_conv_image"] = False


def file_based_input_fn_builder(input_file, is_training, drop_remainder, batch):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    input_file = [os.path.join(input_file, name) for name in os.listdir(input_file)]
    name_to_features = {
        "unid": tf.FixedLenFeature([], tf.int64),
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
        "name": tf.FixedLenFeature([],tf.string)
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


def create_model(input,is_train):
    """Creates a classification model."""
    activate_dict = None
    if model_config["model_name"] == "lib/ResNet":
        model = Model(resnet_size=model_config["resnet_size"], bottleneck=model_config["bottleneck"],
                      num_classes=model_config["num_classes"], num_filters=model_config["num_filters"],
                      kernel_size=model_config["kernel_size"], conv_stride=model_config["conv_stride"],
                      first_pool_size=model_config["first_pool_size"], first_pool_stride=model_config["first_pool_stride"],
                      block_sizes=model_config["block_sizes"], block_strides=model_config["block_strides"],
                      final_size=model_config["final_size"], resnet_version=model_config["resnet_version"],
                      data_format=model_config["data_format"])
    else:
        model = Model(resnet_size=model_config["resnet_size"], bottleneck=model_config["bottleneck"],
                      num_classes=model_config["num_classes"], num_filters=model_config["num_filters"],
                      kernel_size=model_config["kernel_size"], conv_stride=model_config["conv_stride"],
                      first_pool_size=model_config["first_pool_size"], first_pool_stride=model_config["first_pool_stride"],
                      block_sizes=model_config["block_sizes"], block_strides=model_config["block_strides"],
                      final_size=model_config["final_size"], resnet_version=model_config["resnet_version"],
                      data_format=model_config["data_format"])
    cls_layers = model(input, True)
    activate_dict = model.get_activate_dict()
    return cls_layers, activate_dict


def model_fn_builder(learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu=False
                     ):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        train_summary_lt = []
        val_summary_lt = []
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        unids = features["unid"]
        picture = features["image/encoded"]

        if mode == tf.estimator.ModeKeys.TRAIN:
            batch = params["train_batch_size"]
        elif mode == tf.estimator.ModeKeys.EVAL:
            batch = params["eval_batch_size"]
        else:
            batch = params["predict_batch_size"]


        picture_lt = tf.unstack(picture, num=batch)
        if configDir["grad_cam"]:
            organ_image = []
        for i in range(len(picture_lt)):
            image = tf.image.decode_jpeg(picture_lt[i], channels=3)
            image = tf.image.resize_images(image,[configDir["resize"],configDir["resize"]])
            if configDir["grad_cam"]:
                organ_image.append(image)
            picture_lt[i] = tf.image.per_image_standardization(image)
        picture = tf.stack(picture_lt)
        if configDir["grad_cam"]:
            organ_image = tf.stack(organ_image)


        picture = tf.cast(picture, tf.float32)


        # 图像增强
        if mode == tf.estimator.ModeKeys.TRAIN:
            pic_summary1 = tf.summary.image("picture_org", picture)
            train_summary_lt.append(pic_summary1)


            picture = tf.image.random_flip_left_right(picture)
            picture = tf.image.random_flip_up_down(picture)

        label = features["label"]

        cls, activate_dict = create_model(picture,mode == tf.estimator.ModeKeys.TRAIN)

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


        per_example_loss, per_example_probility = Loss.prob_CEloss(cls, label,model_config["num_classes"])
        global_step = tf.train.get_or_create_global_step()

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_hook_lt = []
            per_example_loss_summary = tf.summary.histogram("per_example_loss", -per_example_loss)
            per_example_probility_summary = tf.summary.histogram("per_example_pribility", per_example_probility)
            train_summary_lt.append(per_example_loss_summary)
            train_summary_lt.append(per_example_probility_summary)
            if configDir["loss_name"] == "GHM":
                loss_config = configDir["GHM_loss_config"]
                total_loss,weight = Loss.GHMLoss(-per_example_loss,loss_config["bin"],loss_config["step_length"],batch)
                train_summary_lt.append(tf.summary.histogram("loss_weight",weight))
            elif configDir["loss_name"] == "focal":
                loss_config = configDir["focal_loss_config"]
                total_loss,weight = Loss.focal_loss1(tf.one_hot(label,depth=model_config["num_classes"]),cls,loss_config["gamma"])
                train_summary_lt.append(tf.summary.histogram("loss_weight",weight))
            else:
                total_loss = -tf.reduce_mean(per_example_loss, axis=-1)
            # total_loss = tf.losses.softmax_cross_entropy(tf.one_hot(label,depth=class_dim),cls)

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, -1)
            # train_op = tf.train.AdagradDAOptimizer(learning_rate=configDir["learning_rate"]
            #                                        ,global_step=global_step).minimize(total_loss,global_step=global_step)
            # decay_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,global_step=global_step,decay_steps=configDir["decay_steps"],decay_rate=configDir["decay_rate"])
            # learning_rate_summary = tf.summary.scalar("decay_learning_rate",decay_learning_rate)
            # train_op = tf.train.GradientDescentOptimizer(decay_learning_rate).minimize(total_loss,global_step=global_step)
            summary_hook = tf.train.SummarySaverHook(save_steps=configDir["save_summary_steps"],
                                                     output_dir=configDir["model_dir"]
                                                     , summary_op=train_summary_lt)
            train_hook_lt.append(summary_hook)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss= -tf.reduce_mean(per_example_loss, axis=-1),
                train_op=train_op,
                export_outputs=None,
                training_chief_hooks=None,
                training_hooks=train_hook_lt
                # evalute_hook(handle=handle,feed_handle=test_handle, run_op=total_loss, evl_step=10),
                # train_hook(handle,train_handle)
                # ],

            )
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_hook_lt = []

            def metric_fn(per_example_loss, label, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label, predictions=predictions, weights=None)
                loss = tf.metrics.mean(values=per_example_loss, weights=None)
                return {
                    "eval_accuracy": accuracy,
                    # "eval_loss": loss,
                }

            if configDir["beholder"]:
                activate_lt = []
                for name, v in activate_dict.items():
                    activate_lt.append(v)
                beholder_hook = BeholderHook(configDir["model_dir"], [picture] + activate_lt, np.ndarray([32, 32, 3]))
                eval_hook_lt.append(beholder_hook)
            pic_summary = tf.summary.image("val_picture_org", picture)
            per_example_loss_summary = tf.summary.histogram("val_per_example_loss", per_example_loss)
            per_example_probility_summary = tf.summary.histogram("val_per_example_pribility", per_example_probility)
            val_summary_lt.append(pic_summary)
            val_summary_lt.append(per_example_probility_summary)
            val_summary_lt.append(per_example_loss_summary)
            eval_hook_lt.append(tf.train.SummarySaverHook(save_steps=configDir["save_summary_steps"],
                                                          output_dir=configDir["model_dir"]
                                                          , summary_op=val_summary_lt))
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=-tf.reduce_mean(per_example_loss, axis=-1),
                eval_metric_ops=metric_fn(per_example_loss, label, cls),
                scaffold=scaffold_fn,
                evaluation_hooks=eval_hook_lt
            )

        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "unique_ids": unids,
                "predict": tf.arg_max(cls, -1),
                # "logits": tf.arg_max(tf.nn.softmax(cls, -1), -1),
                "label": label,
                "category_probility": tf.nn.softmax(cls,-1),
                "path": features["name"]
            }
            if configDir["do_save_conv_image"]:
                for k,v in activate_dict.items():
                    predictions[k] = v

            if configDir["grad_cam"]:
                def grad_cam(conv_layer,loss):
                    grads = tf.gradients(loss,conv_layer)[0]
                    norm_grads = tf.nn.l2_normalize(grads,axis=[1,2,3])
                    if model_config["data_format"] == "channels_first":
                        axis = [2,3]
                        axis1 = 1
                    else:
                        axis = [1,2]
                        axis1 = -1
                    weight = tf.reduce_mean(norm_grads,axis=axis,keepdims=True)
                    cams = conv_layer * weight
                    cams = tf.reduce_sum(cams,axis1,keepdims=False)
                    cams = tf.nn.relu(cams)
                    return cams

                conv_layer = activate_dict[configDir["layer_name"]]
                cam = grad_cam(conv_layer,per_example_probility)
                predictions["cam"] = cam
                predictions["image"] = organ_image

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
    if configDir["do_train"] :
        train_examples = len(os.listdir(configDir["train_input"])) * 1000
        num_train_steps = int(
            train_examples / configDir["train_batch_size"] * configDir["num_train_epochs"])
        num_warmup_steps = int(num_train_steps * configDir["warmup_proportion"])

    model_fn = model_fn_builder(
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
            "image/encoded": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        })
        estimator.export_savedmodel(configDir["TFServing_model_path"], serving_input_receiver_fn,
                                    strip_default_attrs=True)
        return 0

    if configDir["do_train"]:
        trainHookLt = []
        evalHookLt = []
        if configDir["debug"]:
            debug_config = configDir["debug_config"]
            if debug_config["tfdbg"]:
                trainHookLt.append(tfdbg.LocalCLIDebugHook())
            elif configDir["tfdbgtensorboard"]:
                trainHookLt.append(tfdbg.TensorBoardDebugHook(grpc_debug_server_addresses="localhost:11111"))

        train_input_fn = file_based_input_fn_builder(input_file=configDir["train_input"], is_training=True, drop_remainder=True,
                                                     batch="train_batch_size")

        # input_files = os.listdir(os.path.join(configDir["DP"], "test"))
        val_input_fn = file_based_input_fn_builder(input_file=configDir["val_input"], is_training=False, drop_remainder=True,
                                                   batch="eval_batch_size")
        trainSpec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps, hooks=trainHookLt)
        valSpec = tf.estimator.EvalSpec(input_fn=val_input_fn, steps=configDir["trainStepVal"],
                                        throttle_secs=configDir["throttle_secs"], hooks=evalHookLt)
        tf.estimator.train_and_evaluate(estimator=estimator, train_spec=trainSpec, eval_spec=valSpec)

    if configDir["do_test"]:
        def filter_conv_image(predict_dic):
            image = {}
            for k,v in predict_dic.items():
                if k.find("conv") != -1 or k.find("block") != -1:
                    image[k] = v
            return image

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Batch size = %d", configDir["test_batch_size"])

        # input_files = os.listdir(os.path.join(configDir["DP"], "test"))
        predict_input_fn = file_based_input_fn_builder(input_file=configDir["predict_input"], is_training=False, drop_remainder=True,
                                                       batch="predict_batch_size")

        wf = open(configDir["test_res_output"], "w", encoding="utf-8")
        for mm, result in enumerate(estimator.predict(
                predict_input_fn, yield_single_examples=True
                , hooks=[
                    # tfdbg.LocalCLIDebugHook(),
                    # tfdbg.TensorBoardDebugHook(grpc_debug_server_addresses="localhost:11111"),
                ]
        )):

            tf.logging.info("Processing example: %d" % (mm))
            #------------临时代码------------------------#
            if mm == 10:
                break
            #------------临时代码------------------------#
            example_id = result["unique_ids"]
            predict = result["predict"]
            label = result["label"]
            category_probility = "_".join([str(i) for i in result["category_probility"].tolist()])
            path = result["path"].decode('utf-8')
            wf.write("{}\t{}\t{}\t{}\t{}\n".format(example_id, predict, label,category_probility,path))
            if configDir["do_save_conv_image"]:
                conv_image = filter_conv_image(result)
                if not os.path.exists(configDir["conv_image_path"]):
                    os.makedirs(configDir["conv_image_path"])
                numpy_path = os.path.join(configDir["conv_image_path"],os.path.basename(path)[:-4])
                np.savez(numpy_path,**conv_image)
            if configDir["grad_cam"]:
                def show_cam_on_image(img, mask):
                    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    cam_out = heatmap + np.float32(img)
                    cam_out = cam_out / np.max(cam_out)
                    plt.imshow(cam_out)
                    plt.show()

                cam = result["cam"]
                image = result["image"]

                cam = cam / np.max(cam)
                cam = cv2.resize(cam, (configDir["resize"], configDir["resize"]))
                image  = image / 255

                # Superimposing the visualization with the image.
                show_cam_on_image(image,cam)

        wf.close()


if __name__ == "__main__":
    tf.app.run()
