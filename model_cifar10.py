import tensorflow as tf


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(phase_train, input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True,   # Use 2x2 max-pooling.
                   use_batch_norm= True):        
    
    # Shape of the filter-weights for the convolution.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution. strides=[1, 2, 2, 1] would mean that the filter is moved 2 pixels across the x- and y-axis of the image. 
    # The padding is set to 'SAME' which means the input image is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    # Add the biases to the results of the convolution. A bias-value is added to each filter-channel.
    layer += biases
    
    # Use batch normalization to reduce covarience shift in the filters. 
    if use_batch_norm:
        layer =  batch_norm(layer, num_filters, phase_train )
    
    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we consider 2x2 windows and select the largest value in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Rectified Linear Unit (ReLU). It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us to learn more complicated functions.
    layer = tf.nn.relu(layer)

    return layer, weights

def batch_norm(x, n_out, phase_train):
    """ 
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        '''
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
 
        def mean_var_with_update():
            #with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
 
        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        '''
        normed = tf.nn.batch_norm_with_global_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3, True)
    return normed


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 phase_train= True,
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def dropout_layer(input, keep_prob):
    layer = tf.nn.dropout(input, keep_prob)
    return layer

# Try adding dropout later
filter_size = 3
num_classes = 10
num_channels= 3

num_filters1=64
num_filters2=64
num_filters3=128
num_filters4=128
num_filters5=256
num_filters6=256
num_filters7=256
num_filters8=512
num_filters9=512
num_filters10=512
num_filters11=512
num_filters12=512
num_filters13=512

fc_size = 512


def load_model(x_image, phase_train, keep_prob):
    layer_conv1, weights_conv1  =       new_conv_layer(phase_train, input=x_image,        num_input_channels=num_channels,   filter_size=filter_size, num_filters=num_filters1,    use_pooling=False,    use_batch_norm= True)
    layer_conv2, weights_conv2  =       new_conv_layer(phase_train, input=layer_conv1,    num_input_channels=num_filters1,   filter_size=filter_size, num_filters=num_filters2,    use_pooling=True,     use_batch_norm = True)
    layer_conv3, weights_conv3  =       new_conv_layer(phase_train, input=layer_conv2,    num_input_channels=num_filters2,   filter_size=filter_size, num_filters=num_filters3,    use_pooling=False,    use_batch_norm = True)
    layer_conv4, weights_conv4  =       new_conv_layer(phase_train, input=layer_conv3,    num_input_channels=num_filters3,   filter_size=filter_size, num_filters=num_filters4,    use_pooling=True,     use_batch_norm = True)
    layer_conv5, weights_conv5  =       new_conv_layer(phase_train, input=layer_conv4,    num_input_channels=num_filters4,   filter_size=filter_size, num_filters=num_filters5,    use_pooling=False,    use_batch_norm = True)
    layer_conv6, weights_conv6  =       new_conv_layer(phase_train, input=layer_conv5,    num_input_channels=num_filters5,   filter_size=filter_size, num_filters=num_filters6,    use_pooling=False,     use_batch_norm = True)
    layer_conv7, weights_conv7  =       new_conv_layer(phase_train, input=layer_conv6,    num_input_channels=num_filters6,   filter_size=filter_size, num_filters=num_filters7,    use_pooling=True,     use_batch_norm = True)
    layer_conv8, weights_conv8  =       new_conv_layer(phase_train, input=layer_conv7,    num_input_channels=num_filters7,   filter_size=filter_size, num_filters=num_filters8,    use_pooling=False,     use_batch_norm = True)
    layer_conv9, weights_conv9   =      new_conv_layer(phase_train, input=layer_conv8,    num_input_channels=num_filters8,   filter_size=filter_size, num_filters=num_filters9,    use_pooling=False,     use_batch_norm = True)
    layer_conv10, weights_conv10 =      new_conv_layer(phase_train, input=layer_conv9,    num_input_channels=num_filters9,   filter_size=filter_size, num_filters=num_filters10,   use_pooling=True,     use_batch_norm = True)
    layer_conv11, weights_conv11 =      new_conv_layer(phase_train, input=layer_conv10,   num_input_channels=num_filters10,  filter_size=filter_size, num_filters=num_filters11,   use_pooling=False,     use_batch_norm = True)
    layer_conv12, weights_conv12 =      new_conv_layer(phase_train, input=layer_conv11,   num_input_channels=num_filters11,  filter_size=filter_size, num_filters=num_filters12,   use_pooling=False,     use_batch_norm = True)
    layer_conv13, weights_conv13 =      new_conv_layer(phase_train, input=layer_conv12,   num_input_channels=num_filters12,  filter_size=filter_size, num_filters=num_filters13,   use_pooling=True,     use_batch_norm = True)
    layer_flat, num_features = flatten_layer(layer_conv13)
    #layer_flat = dropout_layer(layer_flat, keep_prob)
    layer_fc1 = new_fc_layer(input=layer_flat,  num_inputs=num_features, num_outputs=fc_size, phase_train=True, use_relu=True)       
    #layer_fc1  = dropout_layer(layer_fc1, keep_prob) 
    layer_fc2 = new_fc_layer(input=layer_fc1,  num_inputs=fc_size, num_outputs=num_classes, phase_train=True, use_relu=True)
    y_pred = tf.nn.softmax(layer_fc2)       
    
    return y_pred
