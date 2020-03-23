# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves:
Supported layers can be found by running `self.plugin.get_supported_layers(network)`
Then these layers can be compared by network layers (`self.network.layers.keys()`)

      

Some of the potential reasons for handling custom layers are:
  - Underlaying hardware does not support running these layers
  - Intel has already provided implementation for those layers according to the device:
    If any un supported layers found these can be resolved by adding CPU extension:
      `inference_core.add_extension(CPU_EXTENSION,"CPU")`

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were:
  - I ran inference for object detection on yolov3, it took 21.131138 seconds

The difference between model accuracy pre- and post-conversion was :
  - couldn't benchmark this

The size of the model pre- and post-conversion was:
  Model = bvlc_reference_caffenet pre=243.9 MB  post=243.9 MB
  Moel = ResNet101_DUC_HDC pre=260.7 MB  post=259.9 MB

The inference time of the model pre- and post-conversion was:
  - couldn't benchmark this

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
- Counting people in a specific Place
- Predicting a product popularity in stores

Each of these use cases would be useful because:
- Counting people in a specific Place : For example we can check how many people are going to a specific event or place, enabling us to understand behavior of user about some place or event.
- Predicting a product popularity in stores : For example a product's popularity can be rated according how many poeple visit that product place in a store
## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:

- couldn't benchmark this

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: bvlc_reference_caffenet
  - https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet
  - I converted the model to an Intermediate Representation with the following arguments: python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model models/bvlc_reference_caffenet/model.onnx
  - The model was insufficient for the app because it was not able to perform the inference accurately
  
- Model 2: ResNet101_DUC_HDC
  - https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/duc
  - I converted the model to an Intermediate Representation with the following arguments : python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model models/ResNet101_DUC_HDC.onnx
  - The model was insufficient for the app because it took too much time to perform inference, around 3900-4000 ms

- Model 3: [yolo_v3]
  - I converted the model to an Intermediate Representation with the following arguments :  python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model yolo_v3.pb  --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json
  - Was not able to convert:  Error:
    [ ERROR ]  Shape [ -1 416 416   3] is not fully defined for output 0 of "Placeholder". Use --input_shape with positive integers to override model input shapes.
    [ ERROR ]  Cannot infer shapes or values for node "Placeholder".
    [ ERROR ]  Not all output shapes were inferred or fully defined for node "Placeholder".
    For more information please refer to Model Optimizer FAQ (https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Model_Optimizer_FAQ.html), question #40.
    [ ERROR ]
    [ ERROR ]  It can happen due to bug in custom shape infer function <function Parameter.__init__.<locals>.<lambda> at 0x1491844d0>.

