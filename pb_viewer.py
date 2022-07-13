# import tensorflow as tf
#
# from tensorflow.python.platform import gfile
# with tf.Session() as sess:
#     model_filename ='./saved_models/nilesh_test/saved_model.pb'
#     with gfile.FastGFile(model_filename, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         g_in = tf.import_graph_def(graph_def)
# LOGDIR='/pb_viewer'
# train_writer = tf.summary.FileWriter(LOGDIR)
# train_writer.add_graph(sess.graph)
# train_writer.flush()
# train_writer.close()


import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

with tf.Session() as sess:
    model_filename = './saved_models/nilesh_test-1657688437/saved_model.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        data = compat.as_bytes(f.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)
        # print(sm)
        if 1 != len(sm.meta_graphs):
            print('More than one graph found. Not sure which to write')
            sys.exit(1)

    # graph_def = tf.GraphDef()
    # graph_def.ParseFromString(sm.meta_graphs[0])
g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
LOGDIR = './pb_viewer'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
