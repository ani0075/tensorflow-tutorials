import tensorflow as tf
from . import model_utils as mu

class AlexNet(object):
    """
    AlexNet model using CNN layers for the task of image classification on
    ImageNet dataset.

    Parameters
    ----------
    num_classes: int
        Number of classes (1000 for ImageNet).
    scope: str
        Name scope of the model.                # Why?
    """

    def __init__(self, num_classes, scope):
        self.num_classes = num_classes
        self.scope = scope
        self.__debug__tensors = {}              # Why?


    def get_model_fn(self):
        """
        Creates the model function pertaining to the 'Estimator' class
        interface.

        Returns
        -------
        model_fn: callable
            The model function with the following signature:
            model_fn(features, labels, mode, params)
        """
        def model_fn(features, labels, mode, params):
            """
            Parameters
            ----------
            features: Tensor
                A batch of images of shape `(batch size, image height, image
                width, num channels)`
            labels: Tensor
                If mode is ModeKeys.INFER, `labels=None` will be passed.
            mode: tf.estimator. ModeKeys
                Specifies if this is training, evaluation or prediction.
            params: dict
                Optional dictionary of hyperparameters. Will receive what is
                passed to Estimator in params. This allows to configure
                Estimators for hyperparameter tuning.

            Returns
            -------
            predictions: Tensor
                Predictions of the network for input features.
            loss: Tensor
                Loss of the network for the given input features and labels.
            train_op: TensorOp
                The training operation that when run in a session, will update
                model parameters, given input features and labels.
            """
            if mode == tf.estimator.ModeKeys.PREDICT:
                logits, predictions = self.create_model_graph(
                    images_var=features,
                    labels_var=labels,
                    mode=mode)

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={'label':predictions})          # Why?

            else:
                predictions, loss = self.create_model_graph(
                    images_var=features,
                    labels_var=labels,
                    mode=mode)

                train_op = self.get_train_func(
                    loss=loss,
                    learning_rate=params['learning_rate'],
                    mode=mode)

                eval_metric_ops = {
                    'evalmetric/accuracy':
                        tf.contrib.metrics.streaming_accuracy(          # Why?
                            predictions=predictions, labels=labels)
                }

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    loss=loss,
                    train_op=train_op,
                    eval_metric_ops=eval_metric_ops)

        return model_fn

    def create_model_graph(self, images_var, labels_var, mode):
        """
        Create the main computational graph of the model

        Parameters
        ----------
        images_var: Tensor
            placeholder (or variable) for images of shape `(batch size, image
            height, image width, num channels)`
        labels_var: Tensor
            placeholder (or variable) for the class label of the image, of shape
            `(batch size, )
        mode: tf.estimator.ModeKeys
            Run mode for creating the computational graph
        """

        with tf.variable_scope(self.scope, "AlexNet"):
            with tf.variable_scope("conv1"):
                wconv1 = mu.weight([11, 11, 3, 96], name='wconv1')
                bconv1 = mu.bias(0.0, [96], name='bconv1')
                conv1 = tf.add(mu.conv2d(images_var, wconv1, stride=(4,4),
                    padding='SAME'), bconv1)
                conv1 = mu.relu(conv1)
                pool1 = mu.max_pool2d(conv1, kernel_size=[1, 3, 3, 1],
                    stride=[1, 2, 2, 1], padding='VALID')
                tf.summary.histogram('wconv1', wconv1)
                tf.summary.histogram('bconv1', bconv1)

            with tf.variable_scope("conv2"):
                wconv2 = mu.weight([5, 5, 96, 256], name='wconv2')
                bconv2 = mu.bias(1.0, [256], name='bconv2')
                conv2 = tf.add(mu.conv2d(pool1, wconv2, stride=(1,1),
                    padding='SAME'), bconv2)
                conv2 = mu.relu(conv2)
                pool2 = mu.max_pool2d(conv2, kernel_size=[1, 3, 3, 1],
                    stride=[1, 2, 2, 1], padding='VALID')
                tf.summary.histogram('wconv2', wconv2)
                tf.summary.histogram('bconv2', bconv2)

            with tf.variable_scope("conv3"):
                wconv3 = mu.weight([3, 3, 256, 384], name='wconv3')
                bconv3 = mu.bias(0.0, [384], name='bconv3')
                conv3 = tf.add(mu.conv2d(pool2, wconv3, stride=(1,1),
                    padding='SAME'), bconv3)
                conv3 = mu.relu(conv3)
                tf.summary.histogram('wconv3', wconv3)
                tf.summary.histogram('bconv3', bconv3)

            with tf.variable_scope("conv4"):
                wconv4 = mu.weight([3, 3, 384, 384], name='wconv4')
                bconv4 = mu.bias(1.0, [384], name='bconv4')
                conv4 = tf.add(mu.conv2d(conv3, wconv4, stride=(1,1),
                    padding='SAME'), bconv4)
                conv4 = mu.relu(conv4)
                tf.summary.histogram('wconv4', wconv4)
                tf.summary.histogram('bconv4', bconv4)

            with tf.variable_scope("conv5"):
                wconv5 = mu.weight([3, 3, 384, 256], name='wconv5')
                bconv5 = mu.bias(1.0, [256], name='bconv5')
                conv5 = tf.add(mu.conv2d(conv4, wconv5, stride=(1,1),
                    padding='SAME'), bconv5)
                conv5 = mu.relu(conv5)
                pool5 = mu.max_pool2d(conv5, kernel_size=[1, 3, 3, 1],
                    stride=[1, 2, 2, 1], padding='VALID')
                tf.summary.histogram('wconv5', wconv5)
                tf.summary.histogram('bconv5', bconv5)


            pool5_rshp = tf.reshape(tensor=pool5,
                                    shape=[tf.shape(pool5)[0], -1])


            with tf.variable_scope("fc1"):
                wfc1 = mu.weight([6*6*256, 4096], name='wfc1')
                bfc1 = mu.bias(0.0, [4096], name='bfc1')
                fc1 = tf.add(tf.matmul(pool5_rshp, wfc1), bfc1)
                fc1 = mu.relu(fc1)
                fc1 = tf.nn.dropout(fc1, 0.5)
                tf.summary.histogram('wfc1', wfc1)
                tf.summary.histogram('bfc1', bfc1)

            with tf.variable_scope("fc2"):
                wfc2 = mu.weight([4096, 4096], name='wfc2')
                bfc2 = mu.bias(0.0, [4096], name='bfc2')
                fc2 = tf.add(tf.matmul(fc1, wfc2), bfc2)
                fc2 = mu.relu(fc2)
                fc2 = tf.nn.dropout(fc2, 0.5)
                tf.summary.histogram('wfc2', wfc2)
                tf.summary.histogram('bfc2', bfc2)

            with tf.variable_scope("output"):
                wfc3 = mu.weight([4096, 1000], name='wfc3')
                bfc3 = mu.bias(0.0, [1000], name='bfc3')
                logits = tf.add(tf.matmul(fc2, wfc3), bfc3)
                # softmax = tf.nn.softmax(logits)
                tf.summary.histogram('logits', logits)


            predictions = tf.argmax(logits, -1)                 # Why -1?

            if mode != tf.estimator.ModeKeys.PREDICT:
                with tf.variable_scope("loss"):
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels_var,
                        logits=logits)
                    loss = tf.reduce_mean(losses, name='loss')
                tf.summary.scalar('loss', loss)

                with tf.variable_scope("accuracy"):
                    accuracy = tf.contrib.metrics.accuracy(
                        predictions, labels_var)
                tf.summary.scalar('accuracy', accuracy)

                return predictions, loss

            else:
                return logits, predictions


    def get_train_func(self, loss, learning_rate, mode):
        """
        Create the training function for the model.

        Parameters
        ----------
        loss: Tensor
            Tensor variable for the network loss
        learning_rate: float
            Learning rate value
        mode: tf.contrib.learn.ModeKeys
            Specifies if this is training, evaluation, or prediction.

        Returns
        -------
        train_op
        """
        if mode != tf.estimator.ModeKeys.TRAIN or loss is None:
            return None


        global_step = tf.train.get_or_create_global_step()


        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer='Adam',
            summaries=['gradients'])

        return train_op



