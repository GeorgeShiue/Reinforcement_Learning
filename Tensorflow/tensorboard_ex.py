import tensorflow as tf
tf.compat.v1.disable_eager_execution()

with tf.name_scope("Computation"): #建立作用域以區分不同區塊的運算
    with tf.name_scope("part1"):
        a = tf.constant(5)
        b = tf.constant(4)
        c = tf.multiply(a,b)
    with tf.name_scope("part2"):
        d = tf.constant(2)
        e = tf.constant(3)
        f = tf.multiply(d,e)
with tf.name_scope("Result"):
    g = tf.add(c,f)

with tf.compat.v1.Session() as sess:
    writer = tf.compat.v1.summary.FileWriter("output_", sess.graph) #寫入檔案至output_資料夾中
    print(sess.run(g))
    writer.close()