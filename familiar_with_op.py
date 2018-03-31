import tensorflow as tf
import numpy as np
'''
design network
d_x = 3
d_y = 3
imageIn = tf.placeholder(shape=[None, d_x,d_y], dtype=tf.float32)
state_mask = tf.placeholder(shape=[None, d_x,d_y], dtype=tf.float32)
focus_mask = tf.placeholder(shape=[None, d_x,d_y], dtype=tf.float32)
imageIn = tf.reshape(imageIn, shape=[-1, d_x, d_y, 1])
state_mask = tf.reshape(state_mask, shape=[-1, d_x, d_y, 1])
focus_mask = tf.reshape(focus_mask,shape=[-1,d_x,d_y,1])
real_AB = tf.concat([imageIn, state_mask, focus_mask], 3)
'''
'''
#try ops
a = tf.constant(100,dtype = tf.float16)
b = 16*tf.sigmoid(a)-10
sess = tf.InteractiveSession()
#tf.global_variables_initializer().run()
b_out = sess.run(b)
print("b_out",b_out)
aaa = 1
'''






#operation libraries
'''
aaa = 1

a = np.ones((10,10))
a[5,7] = 100
x = tf.placeholder(tf.float32, [10, 10])
y =  tf.constant(3)
z = tf.constant(4)
out = tf.maximum(y,z)
result, indices = tf.nn.top_k(x,k=1,sorted=True,name=None)#tf.argmax(x,name=None)
result = tf.reshape(result, shape = [10,])
indices = tf.reshape(indices,shape=[10,])
index_x = tf.argmax(result)
st = tf.constant([1,2,3])
sum = tf.reduce_sum(st)
it = tf.equal(1,1)
const_v = tf.constant(-1.0, shape=[2, 3])
const_out = const_v * 5
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
res,ind,indx = sess.run((result,indices,index_x), feed_dict={x: a})
o = sess.run(out)
itr = sess.run(it)
sumr = sess.run(sum)
co = sess.run(const_out)
print(o)
print(res)
print(ind)
print("result")
print([indx,ind[indx]])
print("logic")
print(itr)
print("sum")
print(sumr)
print("co")
print(co)'''

X = tf.placeholder("float", [None, 5])
o = tf.ones(shape=tf.shape(X))
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
feed_X = np.ones([3,5])
oo = sess.run((o), feed_dict={X:feed_X })
print(oo)

