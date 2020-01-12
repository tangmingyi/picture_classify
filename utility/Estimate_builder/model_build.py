import tensorflow as tf
from lib.ResNet.model import Model
from lib.ResNet import optimization,myhook,Loss
from lib.ResNet.myhook import BeholderHook
import numpy as np



def create_model(input,is_train,model_config):
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


def grad_cam(conv_layer, loss,data_format):
    grads = tf.gradients(loss, conv_layer)[0]
    norm_grads = tf.nn.l2_normalize(grads, axis=[1, 2, 3])
    if data_format == "channels_first":
        axis = [2, 3]
        axis1 = 1
    else:
        axis = [1, 2]
        axis1 = -1
    weight = tf.reduce_mean(norm_grads, axis=axis, keepdims=True)
    cams = conv_layer * weight
    cams = tf.reduce_sum(cams, axis1, keepdims=False)
    cams = tf.nn.relu(cams)
    return cams

def model_fn_builder(config,model_config,learning_rate,
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

        if config["file_base"]:
            picture_lt = tf.unstack(picture, num=batch)
            if config["grad_cam"]:
                organ_image = []
            for i in range(len(picture_lt)):
                image = tf.image.decode_jpeg(picture_lt[i], channels=3)
                image = tf.image.resize_images(image,[config["resize"],config["resize"]])
                if config["grad_cam"]:
                    organ_image.append(image)
                picture_lt[i] = tf.image.per_image_standardization(image)
            picture = tf.stack(picture_lt)
            if config["grad_cam"]:
                organ_image = tf.stack(organ_image)


            picture = tf.cast(picture, tf.float32)


        # 图像增强
        if mode == tf.estimator.ModeKeys.TRAIN:
            pic_summary1 = tf.summary.image("picture_org", picture)
            train_summary_lt.append(pic_summary1)


            picture = tf.image.random_flip_left_right(picture)
            picture = tf.image.random_flip_up_down(picture)

        label = features["label"]

        cls, activate_dict = create_model(picture,mode == tf.estimator.ModeKeys.TRAIN,model_config=model_config)

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
            if config["loss_name"] == "GHM":
                loss_config = config["GHM_loss_config"]
                total_loss,weight = Loss.GHMLoss(-per_example_loss,loss_config["bin"],loss_config["step_length"],batch)
                train_summary_lt.append(tf.summary.histogram("loss_weight",weight))
            elif config["loss_name"] == "focal":
                loss_config = config["focal_loss_config"]
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
            summary_hook = tf.train.SummarySaverHook(save_steps=config["save_summary_steps"],
                                                     output_dir=config["model_dir"]
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

            if config["beholder"]:
                activate_lt = []
                for name, v in activate_dict.items():
                    activate_lt.append(v)
                beholder_hook = BeholderHook(config["model_dir"], [picture] + activate_lt, np.ndarray([32, 32, 3]))
                eval_hook_lt.append(beholder_hook)
            pic_summary = tf.summary.image("val_picture_org", picture)
            per_example_loss_summary = tf.summary.histogram("val_per_example_loss", per_example_loss)
            per_example_probility_summary = tf.summary.histogram("val_per_example_pribility", per_example_probility)
            val_summary_lt.append(pic_summary)
            val_summary_lt.append(per_example_probility_summary)
            val_summary_lt.append(per_example_loss_summary)
            eval_hook_lt.append(tf.train.SummarySaverHook(save_steps=config["save_summary_steps"],
                                                          output_dir=config["model_dir"]
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
            if config["do_save_conv_image"]:
                for k,v in activate_dict.items():
                    predictions[k] = v

            if config["grad_cam"]:
                conv_layer = activate_dict[config["layer_name"]]
                cam = grad_cam(conv_layer,per_example_probility,model_config["data_format"])
                predictions["cam"] = cam
                predictions["image"] = organ_image

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions, scaffold=scaffold_fn)
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn