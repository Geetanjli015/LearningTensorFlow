import tensorflow as tf

x1=tf.constant(5)
x2=tf.constant(6)

#result=x1*x2
result=tf.multiply(x1,x2)
print(result)
# sess = tf.Session()
# print(sess.run(result))
# sess.close()

with tf.Session() as sess:
	output = sess.run(result)
	print(output)

print(output)

#so basically we can start the session and run and get the result within that session
# and save it in a pythoon variable for referencing ..but once u r outside session we cannot run sessions
print(sess.run())