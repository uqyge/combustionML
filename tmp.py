# single GPU (baseline)
import tensorflow as tf



# place the initial data on the cpu
with tf.device('/cpu:0'):
    input_data = tf.Variable([[1., 2., 3.],
                              [4., 5., 6.],
                              [7., 8., 9.],
                              [10., 11., 12.]])
    b = tf.Variable([[1.], [1.], [2.]])


inputs = tf.split(input_data, 2)




# compute the result on the 0th gpu
with tf.device('/gpu:0'):
    output = tf.matmul(input_data, b)
    output_split_1 = tf.matmul(inputs[0], b)
    output_split_2 = tf.matmul(inputs[1], b)





# create a session and run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(output))
    print(sess.run(output_split_1))
    print(sess.run(output_split_2))