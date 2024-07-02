###  onnx 转化为 trt 方式

#!/usr/bin/env  bash

onnxmodel=decoder/1/vits-decoder-batch.onnx
trtmodel=decoder/1/vits-decoder-batch.plan

export CUDA_VISIBLE_DEVICES=0
/usr/src/tensorrt/bin/trtexec --onnx=$onnxmodel --saveEngine=$trtmodel  --fp16 \
	--minShapes=z_p:1x192x1,y_mask:1x1x1  \
        --maxShapes=z_p:1x192x5120,y_mask:1x1x5120 --buildOnly 


nnxmodel=prosody/1/prosody.onnx
trtmodel=prosody/1/prosody.plan


export CUDA_VISIBLE_DEVICES=0
/usr/src/tensorrt/bin/trtexec  --onnx=$onnxmodel --saveEngine=$trtmodel   \
	--minShapes=word_input_ids:1x1,expand_length:1x1  \
        --maxShapes=word_input_ids:1x500,expand_length:1x500  --buildOnly
