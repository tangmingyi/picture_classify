import tensorflow as tf
def focal_loss1(y_true,y_pred,gamma=2.0):

    # alpha = tf.constant(alpha, dtype=tf.float32)
    #alpha = tf.constant([[1],[1],[1],[1],[1]], dtype=tf.float32)
    #alpha = tf.constant_initializer(alpha)
    gamma = float(gamma)
    epsilon = 1.e-7
    y_true = tf.cast(y_true, tf.float32)
    # y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -tf.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.multiply(weight, ce)
    loss = tf.reduce_mean(fl)
    return loss,weight

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