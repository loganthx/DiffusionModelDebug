# DiffusionModelDebug
Super simple method of checking if the respective model can overfit a dataset containing only one image copied multiple times

# The Simples Diffusion Model

If you can code a UNET and any kind of suitable AttentionLayer, you can code a DiffusionModel.
our forward_process function takes a batch of shape (batch, color_channels, width, height) and return noised_batch with shape (batch, color_channels, width, height), and noises shaped (batch, 1 , 1, 1)

# The Markov Chain
Easy to code, hard to explain. Whatever the logic, any process that adds random stochastic behavior locally at each timestep is a Markov Chain.
In our context:
```
betha_tensor = torch.linspace(schedule_t0, schedule_tf, steps)
alpha_tensor = 1 - betha_tensor
alpha_hat_tensor = torch.cumprod(alpha_tensor, axis=0)
```

With the noised_batch being obtained as follows:

```
batch_input = torch.rand(batch, color_channels, width, height)
noise = torch.rand_like(batch_input)
noised_batch = alpha_hat_tensor.sqrt() * batch_input + torch.sqrt(1 - alpha_hat_tensor)*noise
```

# Debugging
If our model can reproduce the input from pure noise, then it's a reasonable network to train on a real dataset

# Debugging Results
Our image is a random image taken from a website:
![alt text](https://ibb.co/179nYyW)

Results:
EPOCH 0
![alt text](https://ibb.co/jLN1WRX)

EPOCH 1000
![alt text](https://ibb.co/NmgVTM6)
