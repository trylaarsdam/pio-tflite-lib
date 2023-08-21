# Tensorflow Lite for Microcontrollers

A PlatformIO library with the complete and (As of Aug 20, 2023) up-to-date version of Tensorflow Lite for Microcontrollers.

WARNING: The v1.1.0 library update includes changes to TFLite Micro that **removes** the AllOpsResolver. You must now use the MicroMutableOpsResolver instead. Depending on your firmware this could be a breaking change.

If you want to build this library for yourself, check out my guide: https://dev.toddr.org/tensorflow-for-the-portenta-h7/

## Installation
Install the library via PlatformIO Library Manager.

## Examples
There is one example in the examples/basic-inference folder. This example was built for the Arduino Pro Portenta H7, and as such many devices may not be able to fit this example into memory. It is easy to scale down the `RAM_SIZE` define in that example, and then use your own smaller model. The purpose of having such a large example was to demonstrate that this library supports TFLM's latest features (such as LSTM layers)

If you want a full tutorial check out my [dev blog article](https://dev.toddr.org/tensorflow-for-the-portenta-h7/).

## Usage
You can use this library exactly like the TFLite for Microcontrollers documentation describes, located at https://www.tensorflow.org/lite/microcontrollers. You do not need to specically include this library in your project, you can just include the TFLM headers directly, e.x.:
```cpp
#include "tensorflow/lite/micro/all_ops_resolver.h"
```

## Notes
This library is not officially supported by Google, and is not affiliated with Google in any way. This library was created because I needed to make it for work anyway, and I figured I should share my work. If you have any questions about usage, please open an issue on GitHub or contact me at `support@toddr.org`, but please check if there are any GH issues before posting/emailing me.

## Template project
You can find a template project for this library at [@trylaarsdam/pio-tflite](https://github.com/trylaarsdam/pio-tflite). This project has the complete setup for running a model using TFLM, but it is not a library, you can
just start editing `main.cpp` and go.

## Supported Operations
This library supports the following Tensorflow Lite operations:
```cpp
  AddAbs();
  AddAdd();
  AddAddN();
  AddArgMax();
  AddArgMin();
  AddAssignVariable();
  AddAveragePool2D();
  AddBatchToSpaceNd();
  AddBroadcastArgs();
  AddBroadcastTo();
  AddCallOnce();
  AddCast();
  AddCeil();
  AddCircularBuffer();
  AddConcatenation();
  AddConv2D();
  AddCos();
  AddCumSum();
  AddDepthToSpace();
  AddDepthwiseConv2D();
  AddDequantize();
  AddDetectionPostprocess();
  AddDiv();
  AddElu();
  AddEqual();
  AddEthosU();
  AddExp();
  AddExpandDims();
  AddFill();
  AddFloor();
  AddFloorDiv();
  AddFloorMod();
  AddFullyConnected();
  AddGather();
  AddGatherNd();
  AddGreater();
  AddGreaterEqual();
  AddHardSwish();
  AddIf();
  AddL2Normalization();
  AddL2Pool2D();
  AddLeakyRelu();
  AddLess();
  AddLessEqual();
  AddLog();
  AddLogicalAnd();
  AddLogicalNot();
  AddLogicalOr();
  AddLogistic();
  AddLogSoftmax();
  AddMaxPool2D();
  AddMaximum();
  AddMean();
  AddMinimum();
  AddMirrorPad();
  AddMul();
  AddNeg();
  AddNotEqual();
  AddPack();
  AddPad();
  AddPadV2();
  AddPrelu();
  AddQuantize();
  AddReadVariable();
  AddReduceMax();
  AddRelu();
  AddRelu6();
  AddReshape();
  AddResizeBilinear();
  AddResizeNearestNeighbor();
  AddRound();
  AddRsqrt();
  AddSelectV2();
  AddShape();
  AddSin();
  AddSlice();
  AddSoftmax();
  AddSpaceToBatchNd();
  AddSpaceToDepth();
  AddSplit();
  AddSplitV();
  AddSqrt();
  AddSquare();
  AddSquaredDifference();
  AddSqueeze();
  AddStridedSlice();
  AddSub();
  AddSum();
  AddSvdf();
  AddTanh();
  AddTranspose();
  AddTransposeConv();
  AddUnidirectionalSequenceLSTM();
  AddUnpack();
  AddVarHandle();
  AddWhile();
  AddZerosLike();
```
