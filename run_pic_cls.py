"""
todo: add mixup blinear dcl constract
doing: 卷积特征图可视化
"""
import json
import os
from utility.Estimate_builder import model_input,model_build
import tensorflow as tf
import numpy as np
# from tensorboard.plugins.beholder import BeholderHook
import cv2
import matplotlib.pyplot as plt
from tensorflow.python import debug as tfdbg
from utility.data_tool.autoaugment import CIFAR10Policy

configDir = json.load(open("config_file/config.json", "r", encoding="utf-8"))
model_config = json.load(open(configDir["mode_config"], 'r', encoding='utf-8'))
#todo:只有file_base input_fn 支持grade_cam可视化。
if not configDir["file_base"]:
    configDir["grad_cam"] = False

if configDir["do_save_conv_image"]:
    configDir["test_batch_size"] = 1

if not configDir["do_test"]:
    configDir["do_save_conv_image"] = False


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_out = heatmap * 0.4 + np.float32(img)
    cam_out = cam_out / np.max(cam_out)
    plt.imshow(heatmap)
    plt.show()
    plt.imshow(cam_out)
    plt.show()


def filter_conv_image(predict_dic):
    image = {}
    for k, v in predict_dic.items():
        if k.find("conv") != -1 or k.find("block") != -1:
            image[k] = v
    return image

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

    model_fn = model_build.model_fn_builder(config=configDir,model_config=model_config,
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

        if configDir["file_base"]:
            train_input_fn = model_input.file_based_input_fn_builder(input_file=configDir["train_input"], is_training=True, drop_remainder=True,
                                                         batch="train_batch_size")

            val_input_fn = model_input.file_based_input_fn_builder(input_file=configDir["val_input"], is_training=False, drop_remainder=True,
                                                       batch="eval_batch_size")
        else:
            augment_fn = CIFAR10Policy()
            train_genter_fn = model_input.get_generator_fn(configDir,configDir["train_input"],augment_fn)
            train_input_fn = model_input.input_fn_builder(configDir,train_genter_fn,True,True,"train_batch_size")

            # input_files = os.listdir(os.path.join(configDir["DP"], "test"))
            val_genter_fn = model_input.get_generator_fn(configDir,configDir["val_input"])
            val_input_fn = model_input.input_fn_builder(configDir,val_genter_fn,False,True,"eval_batch_size")


        trainSpec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps, hooks=trainHookLt)
        valSpec = tf.estimator.EvalSpec(input_fn=val_input_fn, steps=configDir["trainStepVal"],
                                        throttle_secs=configDir["throttle_secs"], hooks=evalHookLt)
        tf.estimator.train_and_evaluate(estimator=estimator, train_spec=trainSpec, eval_spec=valSpec)

    if configDir["do_test"]:

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Batch size = %d", configDir["test_batch_size"])

        # input_files = os.listdir(os.path.join(configDir["DP"], "test"))
        if configDir["file_base"]:
            predict_input_fn = model_input.file_based_input_fn_builder(input_file=configDir["predict_input"], is_training=False, drop_remainder=True,
                                                           batch="predict_batch_size")
        else:
            predict_genter_fn = model_input.get_generator_fn(configDir, configDir["predict_input"])
            predict_input_fn = model_input.input_fn_builder(configDir, predict_genter_fn, False, True, "predict_batch_size")


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
