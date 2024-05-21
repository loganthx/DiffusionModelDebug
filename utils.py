import torch, PIL.Image as Image, matplotlib.pyplot as plt, numpy as np, requests
import torchvision.transforms as transforms
from io import BytesIO




def forward_process_on_batch(batch_sample, t, steps, start_t=0.0, end_t=1.0):
	bethas = torch.linspace(start_t, end_t, steps)
	alphas = 1 - bethas
	alpha_hat = torch.cumprod(alphas, axis=0)
	alpha_hat_t = alpha_hat.gather(-1, t).reshape(-1,1,1,1)
	noise = torch.rand_like(batch_sample)
	result = batch_sample*alpha_hat_t.sqrt() + noise*torch.sqrt(1-alpha_hat_t)
	return result, noise

def get_dummy_img(img_size=100):
	link = f'https://picsum.photos/{img_size}'
	response = requests.get(link)
	image = Image.open(BytesIO(response.content))
	return image

def show_numpy_img(img):
	plt.imshow(img); plt.show()

def single_to_batch(img_tensor, batch_size):
	return torch.stack([img_tensor] * batch_size)

def to_batch(sample_list):
	return torch.stack(sample_list)

def get_transforms(target_size=(100,100)):
	default_transform = transforms.Compose([
		transforms.Resize(target_size),
		transforms.ToTensor(),
		transforms.Lambda(lambda x: (2*x-1) )
		])
	reverse_transform = transforms.Compose([
		transforms.Lambda(lambda x: (x+1)/2),
		transforms.Lambda(lambda x: x.permute(1,2,0)),
		transforms.Lambda(lambda x: x * 255.0),
		transforms.Lambda(lambda x: x.numpy().astype(np.uint8)),
		transforms.ToPILImage()
		])
	return default_transform, reverse_transform





def plot_noise_distribution(noise, predicted_noise):
	plt.hist(noise.cpu().numpy().flatten(), density = True, alpha = 0.8, label = "ground truth noise")
	plt.hist(predicted_noise.cpu().numpy().flatten(), density = True, alpha = 0.8, label = "predicted noise")
	plt.legend()
	plt.show()

def plot_noise_prediction(noise, predicted_noise, reverse_transform):
	plt.figure(figsize=(15,15))
	f, ax = plt.subplots(1, 2, figsize = (5,5))
	ax[0].imshow(reverse_transform(noise.cpu()))
	ax[0].set_title(f"ground truth noise", fontsize = 10)
	ax[1].imshow(reverse_transform(predicted_noise.cpu()))
	ax[1].set_title(f"predicted noise", fontsize = 10)
	plt.show()



