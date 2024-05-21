from utils import forward_process_on_batch, get_dummy_img, get_transforms
import matplotlib.pyplot as plt
import torch


pil_img = get_dummy_img()
default_transf, reverse_transf = get_transforms()

tensor_img = default_transf(pil_img)
#plt.imshow(reverse_transf(tensor_img)); plt.show()

batch_size = 15
output, noise = forward_process_on_batch(batch_sample=torch.stack([tensor_img] * batch_size), t=torch.tensor([i for i in range(batch_size)]), steps=batch_size, start_t=0.0, end_t=0.75)
for i in range(batch_size):
	img_tensor = output[i]
	plt.imshow(reverse_transf(img_tensor)); plt.show()



