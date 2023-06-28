# import tensorflow as tf
# from tensorflow.python.compiler.tensorrt import trt_convert as trt
# # FP16
# conversion_params = trt.TrtConversionParams(precision_mode=trt.TrtPrecisionMode.FP16)
# # is_training=False is important to freeze batch norm statistics
# converter = trt.TrtGraphConverterV2(
#     input_saved_model_dir="models/ssd_mobilenet_v2_fpnlite_320x320/saved_model",
#     conversion_params=conversion_params)
# converter.convert()
# converter.save("saved_model_trt_fp16")
# # # FP32
# # conversion_params = trt.TrtConversionParams()
# # converter = trt.TrtGraphConverterV2(
# #     input_saved_model_dir="models/ssd_mobilenet_v2_fpnlite_320x320/saved_model",
# #     conversion_params=conversion_params)
# # converter.convert()
# # converter.save("saved_model_trt_fp32")
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
# FP16
m = tf.saved_model.load("saved_model_trt_fp16")
ff = m.signatures['serving_default']
ff = convert_variables_to_constants_v2(ff)
graph_def = ff.graph.as_graph_def(True)
tf.io.write_graph(graph_def, '.', 'model_trt_fp16.pb', as_text=False)