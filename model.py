import tensorflow.contrib.layers as layers
import tensorflow as tf

def Net(image, joint_num, stage=6):
    with tf.variable_scope('PoseNet'):
        with tf.variable_scope('FeatureExtractor'):
            #   Assuming the input of the image is 368*368
            image_bn = layers.batch_norm(image)
            #   out : 368 * 368 * 64
            conv1_1 = layers.conv2d(
                image_bn, 64, 3, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv1_1')
            conv1_1_bn = tf.nn.relu(layers.batch_norm(conv1_1))
            #   out : 368 * 368 * 64
            conv1_2 = layers.conv2d(
                conv1_1_bn, 64, 3, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv1_2')
            conv1_2_bn = tf.nn.relu(layers.batch_norm(conv1_2))
            #   out : 184 * 184 * 64
            pool1_stage1 = layers.max_pool2d(conv1_2_bn, 2, 2)
            #   out : 184 * 184 * 128
            conv2_1 = layers.conv2d(pool1_stage1, 128, 3, 1,
                                    activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv2_1')
            conv2_1_bn = tf.nn.relu(layers.batch_norm(conv2_1))
            #   out : 184 * 184 * 128
            conv2_2 = layers.conv2d(
                conv2_1_bn, 128, 3, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv2_2')
            conv2_2_bn = tf.nn.relu(layers.batch_norm(conv2_2))
            #   out : 92 * 92 * 256
            pool2_stage1 = layers.max_pool2d(conv2_2_bn, 2, 2)
            #   out : 92 * 92 * 256
            conv3_1 = layers.conv2d(pool2_stage1, 256, 3, 1,
                                    activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv3_1')
            conv3_1_bn = tf.nn.relu(layers.batch_norm(conv3_1))
            #   out : 92 * 92 * 256
            conv3_2 = layers.conv2d(
                conv3_1_bn, 256, 3, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv3_2')
            conv3_2_bn = tf.nn.relu(layers.batch_norm(conv3_2))
            #   out : 92 * 92 * 256
            conv3_3 = layers.conv2d(
                conv3_2, 256, 3, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv3_3')
            conv3_3_bn = tf.nn.relu(layers.batch_norm(conv3_3))
            #   out : 92 * 92 * 256
            conv3_4 = layers.conv2d(
                conv3_3_bn, 256, 3, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv3_4')
            conv3_4_bn = tf.nn.relu(layers.batch_norm(conv3_4))
            #   out : 46 * 46 * 512
            pool3_stage1 = layers.max_pool2d(conv3_4_bn, 2, 2)
            conv4_1 = layers.conv2d(pool3_stage1, 512, 3, 1,
                                    activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv4_1')
            conv4_1_bn = tf.nn.relu(layers.batch_norm(conv4_1))
            conv4_2 = layers.conv2d(
                conv4_1_bn, 512, 3, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv4_2')
            conv4_2_bn = tf.nn.relu(layers.batch_norm(conv4_2))
        with tf.variable_scope('CPM_stage1'):
            conv4_3_CPM = layers.conv2d(
                conv4_2_bn, 256, 3, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv4_3_CPM')
            conv4_3_CPM_bn = tf.nn.relu(layers.batch_norm(conv4_3_CPM))
            conv4_4_CPM = layers.conv2d(
                conv4_3_CPM_bn, 256, 3, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv4_4_CPM')
            conv4_4_CPM_bn = tf.nn.relu(layers.batch_norm(conv4_4_CPM))
            conv4_5_CPM = layers.conv2d(
                conv4_4_CPM_bn, 256, 3, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv4_5_CPM')
            conv4_5_CPM_bn = tf.nn.relu(layers.batch_norm(conv4_5_CPM))
            conv4_6_CPM = layers.conv2d(
                conv4_5_CPM_bn, 256, 3, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv4_6_CPM')
            conv4_6_CPM_bn = tf.nn.relu(layers.batch_norm(conv4_6_CPM))
            conv4_7_CPM = layers.conv2d(
                conv4_6_CPM_bn, 128, 3, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv4_7_CPM')
            conv4_7_CPM_bn = tf.nn.relu(layers.batch_norm(conv4_7_CPM))
            conv5_1_CPM = layers.conv2d(
                conv4_7_CPM_bn, 512, 1, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv5_1_CPM')
            conv5_1_CPM_bn = tf.nn.relu(layers.batch_norm(conv5_1_CPM))
            conv5_2_CPM = layers.conv2d(
                conv5_1_CPM_bn, joint_num+1, 1, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='conv5_2_CPM')
            conv5_2_CPM_bn = layers.batch_norm(conv5_2_CPM,zero_debias_moving_mean=True)
        with tf.variable_scope('CPM_stage2'):
            concat_stage2 = tf.concat(
                [conv5_2_CPM_bn, conv4_4_CPM], 3)
            Mconv1_stage2 = layers.conv2d(
                concat_stage2, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv1_stage2')
            Mconv1_stage2_bn = tf.nn.relu(layers.batch_norm(Mconv1_stage2))
            Mconv2_stage2 = layers.conv2d(
                Mconv1_stage2_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv2_stage2')
            Mconv2_stage2_bn = tf.nn.relu(layers.batch_norm(Mconv2_stage2))
            Mconv3_stage2 = layers.conv2d(
                Mconv2_stage2_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv3_stage2')
            Mconv3_stage2_bn = tf.nn.relu(layers.batch_norm(Mconv3_stage2))
            Mconv4_stage2 = layers.conv2d(
                Mconv3_stage2_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv4_stage2')
            Mconv4_stage2_bn = tf.nn.relu(layers.batch_norm(Mconv4_stage2))
            Mconv5_stage2 = layers.conv2d(
                Mconv4_stage2_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv5_stage2')
            Mconv5_stage2_bn = tf.nn.relu(layers.batch_norm(Mconv5_stage2))
            Mconv6_stage2 = layers.conv2d(
                Mconv5_stage2_bn, 128, 1, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv6_stage2')
            Mconv6_stage2_bn = tf.nn.relu(layers.batch_norm(Mconv6_stage2))
            Mconv7_stage2 = layers.conv2d(
                Mconv6_stage2_bn, joint_num+1, 1, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='Mconv7_stage2')
            Mconv7_stage2_bn = layers.batch_norm(Mconv7_stage2,zero_debias_moving_mean=True)
        with tf.variable_scope('CPM_stage3'):
            concat_stage3 = tf.concat(
                [Mconv7_stage2_bn, conv4_4_CPM], 3)
            Mconv1_stage3 = layers.conv2d(
                concat_stage3, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv1_stage3')
            Mconv1_stage3_bn = tf.nn.relu(layers.batch_norm(Mconv1_stage3))
            Mconv2_stage3 = layers.conv2d(
                Mconv1_stage3_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv2_stage3')
            Mconv2_stage3_bn = tf.nn.relu(layers.batch_norm(Mconv2_stage3))
            Mconv3_stage3 = layers.conv2d(
                Mconv2_stage3_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv3_stage3')
            Mconv3_stage3_bn = tf.nn.relu(layers.batch_norm(Mconv3_stage3))
            Mconv4_stage3 = layers.conv2d(
                Mconv3_stage3_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv4_stage3')
            Mconv4_stage3_bn = tf.nn.relu(layers.batch_norm(Mconv4_stage3))
            Mconv5_stage3 = layers.conv2d(
                Mconv4_stage3_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv5_stage3')
            Mconv5_stage3_bn = tf.nn.relu(layers.batch_norm(Mconv5_stage3))
            Mconv6_stage3 = layers.conv2d(
                Mconv5_stage3_bn, 128, 1, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv6_stage3')
            Mconv6_stage3_bn = tf.nn.relu(layers.batch_norm(Mconv6_stage3))
            Mconv7_stage3 = layers.conv2d(
                Mconv6_stage3_bn, joint_num+1, 1, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='Mconv7_stage3')
            Mconv7_stage3_bn = layers.batch_norm(Mconv7_stage3,zero_debias_moving_mean=True)
        with tf.variable_scope('CPM_stage4'):
            concat_stage4 = tf.concat(
                [Mconv7_stage3_bn, conv4_4_CPM], 3)
            Mconv1_stage4 = layers.conv2d(
                concat_stage4, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv1_stage4')
            Mconv1_stage4_bn = tf.nn.relu(layers.batch_norm(Mconv1_stage4))
            Mconv2_stage4 = layers.conv2d(
                Mconv1_stage4_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv2_stage4')
            Mconv2_stage4_bn = tf.nn.relu(layers.batch_norm(Mconv2_stage4))
            Mconv3_stage4 = layers.conv2d(
                Mconv2_stage4_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv3_stage4')
            Mconv3_stage4_bn = tf.nn.relu(layers.batch_norm(Mconv3_stage4))
            Mconv4_stage4 = layers.conv2d(
                Mconv3_stage4_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv4_stage4')
            Mconv4_stage4_bn = tf.nn.relu(layers.batch_norm(Mconv4_stage4))
            Mconv5_stage4 = layers.conv2d(
                Mconv4_stage4_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv5_stage4')
            Mconv5_stage4_bn = tf.nn.relu(layers.batch_norm(Mconv5_stage4))
            Mconv6_stage4 = layers.conv2d(
                Mconv5_stage4_bn, 128, 1, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv6_stage4')
            Mconv6_stage4_bn = tf.nn.relu(layers.batch_norm(Mconv6_stage4))
            Mconv7_stage4 = layers.conv2d(
                Mconv6_stage4_bn, joint_num+1, 1, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='Mconv7_stage4')
            Mconv7_stage4_bn = layers.batch_norm(Mconv7_stage4,zero_debias_moving_mean=True)
        with tf.variable_scope('CPM_stage5'):
            concat_stage5 = tf.concat(
                [Mconv7_stage4_bn, conv4_4_CPM], 3)
            Mconv1_stage5 = layers.conv2d(
                concat_stage5, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv1_stage5')
            Mconv1_stage5_bn = tf.nn.relu(layers.batch_norm(Mconv1_stage5))
            Mconv2_stage5 = layers.conv2d(
                Mconv1_stage5_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv2_stage5')
            Mconv2_stage5_bn = tf.nn.relu(layers.batch_norm(Mconv2_stage5))
            Mconv3_stage5 = layers.conv2d(
                Mconv2_stage5_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv3_stage5')
            Mconv3_stage5_bn = tf.nn.relu(layers.batch_norm(Mconv3_stage5))
            Mconv4_stage5 = layers.conv2d(
                Mconv3_stage5_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv4_stage5')
            Mconv4_stage5_bn = tf.nn.relu(layers.batch_norm(Mconv4_stage5))
            Mconv5_stage5 = layers.conv2d(
                Mconv4_stage5_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv5_stage5')
            Mconv5_stage5_bn = tf.nn.relu(layers.batch_norm(Mconv5_stage5))
            Mconv6_stage5 = layers.conv2d(
                Mconv5_stage5_bn, 128, 1, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv6_stage5')
            Mconv6_stage5_bn = tf.nn.relu(layers.batch_norm(Mconv6_stage5))
            Mconv7_stage5 = layers.conv2d(
                Mconv6_stage5_bn, joint_num+1, 1, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(),  scope='Mconv7_stage5')
            Mconv7_stage5_bn = layers.batch_norm(Mconv7_stage5,zero_debias_moving_mean=True)
        with tf.variable_scope('CPM_stage6'):
            concat_stage6 = tf.concat(
                [Mconv7_stage5_bn, conv4_4_CPM], 3)
            Mconv1_stage6 = layers.conv2d(
                concat_stage6, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv1_stage6')
            Mconv1_stage6_bn = tf.nn.relu(layers.batch_norm(Mconv1_stage6))
            Mconv2_stage6 = layers.conv2d(
                Mconv1_stage6_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv2_stage6')
            Mconv2_stage6_bn = tf.nn.relu(layers.batch_norm(Mconv2_stage6))
            Mconv3_stage6 = layers.conv2d(
                Mconv2_stage6_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv3_stage6')
            Mconv3_stage6_bn = tf.nn.relu(layers.batch_norm(Mconv3_stage6))
            Mconv4_stage6 = layers.conv2d(
                Mconv3_stage6_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv4_stage6')
            Mconv4_stage6_bn = tf.nn.relu(layers.batch_norm(Mconv4_stage6))
            Mconv5_stage6 = layers.conv2d(
                Mconv4_stage6_bn, 128, 7, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv5_stage6')
            Mconv5_stage6_bn = tf.nn.relu(layers.batch_norm(Mconv5_stage6))
            Mconv6_stage6 = layers.conv2d(
                Mconv5_stage6_bn, 128, 1, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv6_stage6')
            Mconv6_stage6_bn = tf.nn.relu(layers.batch_norm(Mconv6_stage6))
            Mconv7_stage6 = layers.conv2d(
                Mconv6_stage6_bn, joint_num+1, 1, 1,  activation_fn=None,  weights_initializer=layers.xavier_initializer(), 
                scope='Mconv7_stage6')
            Mconv7_stage6_bn = layers.batch_norm(Mconv7_stage6,zero_debias_moving_mean=True)
        last_conv_each_stage = [conv5_2_CPM_bn,
                                Mconv7_stage2_bn,
                                Mconv7_stage3_bn,
                                Mconv7_stage4_bn,
                                Mconv7_stage5_bn,
                                Mconv7_stage6_bn]
    return last_conv_each_stage[:stage]