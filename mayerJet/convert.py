import tensorrt as trt
import model
import sys
import uff
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import keras
from keras import backend as K

sess = K.get_session()

constant_graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        ["name_of_the_output_graph_node"])

graph_io.write_graph(constant_graph, "./model_save",
                     "output_model_name", as_text=False)