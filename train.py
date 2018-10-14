## Phase 1 training code
import tensorflow as tf
import numpy as np
from datapipeline import DataPipeline
from network import AlexNet
import os
from datetime import datetime
#from tensorflow.data import Iterator
from tensorflow.contrib.data import Iterator

###################

filewriter_path = "./tensorboard/"
checkpoint_path = "./checkpoints/"

if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
    
learning_rate = 0.01
num_epochs = 10
batch_size = 16

train_file = './train_details.json'
test_file = './test_details.json'

# Network params
dropout_rate = 0.5

train_layers = ['fc6', 'fc7', 'fc8']

num_classes = [52, 92, 66]

with tf.device('/cpu:0'):
    tr_data = DataPipeline(train_file,mode='training',batch_size=batch_size,shuffle=True)
    val_data = DataPipeline(test_file,mode='inference',batch_size=batch_size,shuffle=True)
    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])

y1 = tf.placeholder(tf.float32, [batch_size, num_classes[0]])
y2 = tf.placeholder(tf.float32, [batch_size, num_classes[1]])
y3 = tf.placeholder(tf.float32, [batch_size, num_classes[2]])

keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, train_layers, num_classes)

score_1 = model.fc_1
score_2 = model.fc_2
score_3 = model.fc_3

with tf.name_scope("cross_entropy"):
    loss1 = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(score_1, y1))
    loss2 = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(score_2, y2))
    loss3 = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(score_3, y3))

loss = loss1 + loss2 + loss3

var_list = tf.trainable_variables()
with tf.name_scope("train"):
    gradients = tf.gradients(loss, var_list)
    grads_and_vars = list(zip(gradients, var_list))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
    

# Add gradients to summary
for gradient, var in grads_and_vars:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
#with tf.name_scope("accuracy"):
    #correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
#tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    print 'Now starting training..'
    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, gt_attr1, gt_attr2, gt_attr3 = sess.run(next_batch)

            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,
                                          y1: gt_attr1,
                                          y2: gt_attr2,
                                          y3: gt_attr3,
                                          keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y1: gt_attr1,
                                                        y2: gt_attr2,
                                                        y3: gt_attr3,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch*train_batches_per_epoch + step)

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))


