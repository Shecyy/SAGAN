import numpy as np
import onnxruntime
import torch
from PIL import Image
import torchvision.transforms as transforms

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
