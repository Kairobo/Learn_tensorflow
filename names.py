import tensorflow as tf
a = tf.constant(1)
print(a.name)  # prints "Const:0"

b = tf.Variable(1)
print(b.name)  # prints "Variable:0"
a1 = tf.constant(1, name="a")
print(a1.name)  # prints "b:0"

b1 = tf.Variable(1, name="b")
print(b1.name)  # prints "b:0"
with tf.variable_scope('Text_Network'):
    c = tf.get_variable(shape=[],name="c")
    print(c.name)