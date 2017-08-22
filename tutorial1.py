import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def main():
    #declaration
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0) # also tf.float32 implicitly
    print(node1)
    print(node2)
    #session
    sess = tf.Session()
    print(sess.run([node1, node2]))
    node3 = tf.add(node1, node2)
    print("node3:", node3)
    print("sess.run(node3):", sess.run(node3))

    a = tf.placeholder(tf.float32) # similar to float a
    b = tf.placeholder(tf.float32)
    sum_node = a+b #nodes can be added using the Symbol too
    thrice = sum_node*3;
    print(tf.Session().run(sum_node, {a: 4, b: 5.2})) #passing single values
    print(sess.run(sum_node, {a: [2, 3.8], b: [1.9, 3.6]})) #passing an array of values

    print(sess.run(thrice, {a: 2, b: 3})) #complex function
    #complex function execution requires only final call and not all function calls

    #variables
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    #not printable?!

    #initialising a variable
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

    #loss, or checking how good a model is
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

main()
