# SAGAN
SAGAN: Deep Semantic-Aware Generative Adversarial Network for Unsupervised Image Enhancement

This paper has been submitted to Artificial Intelligence Review. The code will be made public when our paper is accepted.

## Requirements
*  Python 3.7.13
*  Torch 1.12.0
*  Visdom 0.1.8.9
*  Torchvision 0.13.0
*  Numpy 1.21.6
*  Pillow 9.2.0
*  Onnx 1.13.1
*  Onnxruntime 1.13.1

## Datasets
*  Training dataset: [Unpaired images](https://drive.google.com/drive/folders/1fwqz8-RnTfxgIIkebFG2Ej3jQFsYECh0)
*  Testing dataset: [MEF, LIME, NPE, DICM]()

## Model
Download SAGAN model from [Inference model]()

## Usage
```python
import numpy as np
import onnxruntime

input_img = Image.open('test.png').convert('RGB')
transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)
input_img = transform(input_img)
input_img = torch.unsqueeze(input_img, 0).numpy()

inference = onnxruntime.InferenceSession('model.onnx')
input_img = {'input': input_img}
output_image = inference.run(['output'], input_img)[0]

output_image = (np.transpose(output_image[0], (1, 2, 0)) + 1) / 2.0 * 255
output_image = np.clip(output_image, 0, 255).astype(np.uint8)
output_image = Image.fromarray(output_image)
output_image.save('result.png')
``` 
