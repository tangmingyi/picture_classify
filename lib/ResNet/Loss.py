import tensorflow as tf
def focal_loss1(y_true,logits,gamma=2.0):
    """
    L_focal = (1-p)^/alph * log(p)
    :param y_true: onehot
    :param logits:
    :param gamma:
    :return:
    """
    gamma = float(gamma)
    y_true = tf.cast(y_true, tf.float32)
    probility = tf.reduce_sum(tf.nn.softmax(logits, axis=-1)*y_true,axis=-1)
    weight = tf.pow(tf.subtract(tf.ones(tf.shape(probility),dtype=tf.float32), probility), gamma)
    probility = tf.clip_by_value(probility, 1e-8, 1.0) #防止log的输入为负值，因此进行裁剪
    loss = tf.reduce_mean(weight*(tf.log(probility)))
    return -loss,weight

def GHMLoss(per_example_loss, bin, step_length, batch):
    edges = [step_length * i for i in range(bin + 1)]
    wight = tf.ones(batch)*(1.0/float(batch))
    for i in range(bin):
        inds = tf.greater_equal(per_example_loss, tf.ones(batch) * edges[i]) & tf.less(per_example_loss,tf.ones(batch) * edges[i + 1])
        num_in_bin = tf.reduce_sum(tf.cast(inds,tf.float32),axis=0)
        wight_in_bin = tf.divide(1.0,num_in_bin)
        wight = tf.where(inds,tf.ones(batch)*wight_in_bin,wight)
    loss = tf.reduce_mean(per_example_loss * wight,axis=0)
    return loss,wight

def prob_CEloss(logits, positions,class_dim):
    one_hot_positions = tf.one_hot(
        positions, depth=class_dim, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    probility = tf.reduce_sum(one_hot_positions * tf.nn.softmax(logits, axis=-1), axis=-1)
    loss = tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
    return loss, probility