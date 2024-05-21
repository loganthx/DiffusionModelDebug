from UNET import UNet
from DiffusionModel import DiffusionModel
from utils import get_dummy_img, get_transforms, plot_noise_prediction, plot_noise_distribution
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision


device = "cuda" if torch.cuda.is_available() else "cpu"
# device='cpu'
print(f"Running on {device} device")



IMAGE_SHAPE = (32,32)
pil_img = get_dummy_img()
print("Test Image:")
plt.imshow(np.asarray(pil_img));plt.show()
default_transform, reverse_transform = get_transforms(target_size=IMAGE_SHAPE)
torch_img = default_transform(pil_img)
LR=0.01
BATCH_SIZE=128
EPOCHS=1001
VERBOSE = 1
PRINT_FREQUENCY = 500
sample_batch = torch.stack([torch_img] * BATCH_SIZE)

diffusion_model = DiffusionModel()
model = UNet(labels=False)
model.to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

for epoch in range(EPOCHS):
	t = torch.randint(0, diffusion_model.timesteps, (BATCH_SIZE,)).long().to(device)
	batch_noisy, noise = diffusion_model.forward(sample_batch, t, device) 
	predicted_noise = model(batch_noisy, t)
	#print(f"predicted_noise shape: {predicted_noise.shape}")
	optimizer.zero_grad()
	loss = loss_function(predicted_noise, noise)
	loss.backward()
	optimizer.step()
	print(f"Epoch: {epoch} Loss: {loss.item()}")
	if epoch % PRINT_FREQUENCY == 0:
		if VERBOSE:
			with torch.no_grad():
				plot_noise_prediction(noise[0], predicted_noise[0], reverse_transform)
				plot_noise_distribution(noise, predicted_noise)


with torch.no_grad():
	img = torch.randn((1, 3) + IMAGE_SHAPE).to(device)
	for i in reversed(range(diffusion_model.timesteps)):
		t = torch.full((1,), i, dtype=torch.long, device=device)
		img = diffusion_model.backward(img, t, model.eval())
		if i % 50 == 0:
			plt.figure(figsize=(2,2))
			plt.imshow(reverse_transform(img[0].cpu()))
			plt.show()










