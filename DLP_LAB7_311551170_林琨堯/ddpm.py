from dataloader import CustomDataset, iclevrLoader
# from dataloader_b import iclevrLoader
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms as transforms
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from torchvision.utils import save_image, make_grid
from dataset.evaluator  import evaluation_model
#%%
train_data_file = 'train.json'
test_data_file = 'test.json'
new_test_data_file = 'new_test.json'
label_file = 'objects.json'


# =============================================================================
# train_dataset = CustomDataset(train_data_file, mode="train")
# test_dataset = CustomDataset(test_data_file, mode="test")
# new_test_dataset = CustomDataset(new_test_data_file, mode="test")
# =============================================================================

train_dataset = iclevrLoader('dataset', mode="train")
test_dataset = iclevrLoader('dataset', mode="test")
new_test_dataset = iclevrLoader('dataset', mode="new_test")


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
new_test_dataloader = DataLoader(new_test_dataset, batch_size=32, shuffle=False)


#%%
def save_images(images, name,epoch):
    #print(images[0])
    grid = torchvision.utils.make_grid(images)
    save_image(grid, fp = "./img/"+name+"/"+epoch+".png")

def sample(model, device, test_loader, filename,epoch):
    # denormalize
    transform=transforms.Compose([
            transforms.Normalize((0, 0, 0), (1/0.5, 1/0.5, 1/0.5)),
            transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
        ])
    model.eval()
    xt = torch.randn(32, 3, 64, 64).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    with torch.no_grad():
        for batch_idx, (img, cond) in enumerate(test_loader):
            xt = xt.to(device,dtype=torch.float32)
            cond = cond.to(device,dtype=torch.float32)
            cond = cond.squeeze()
            #print(cond)
            for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
                # Get model pred
                with torch.no_grad():
                    residual = model(sample = xt, timestep = t, class_labels = cond)  # Again, note that we pass in our labels y
            
                # Update sample with step
                xt = noise_scheduler.step(residual.sample, t, xt).prev_sample
                
            
            
            # evaluate, only accept one-hot
            evaluate = evaluation_model()
            acc = evaluate.eval(xt.to(device), cond.to(device))
            imgg = transform(xt)
            print("Test Result:", acc)

            # save_images(imgg, filename,epoch)
            return acc
#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
# =============================================================================
# class ClassConditionedUnet(nn.Module):
#   def __init__(self, num_classes=25, class_emb_size=1):
#     super().__init__()
#     
#     # The embedding layer will map the class label to a vector of size class_emb_size
#     print(num_classes)
#     self.class_emb = nn.Embedding(num_classes, class_emb_size)
# 
#     # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
#     self.model = UNet2DModel(
#         sample_size=64,           # the target image resolution
#         in_channels=3 + class_emb_size*3, # Additional input channels for class cond.
#         out_channels=3,           # the number of output channels
#         layers_per_block=2,       # how many ResNet layers to use per UNet block
#         block_out_channels = (128, 128, 256, 256, 512, 512), 
#         down_block_types=(
#         "DownBlock2D",  # a regular ResNet downsampling block
#         "DownBlock2D",
#         "DownBlock2D",
#         "DownBlock2D",
#         "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
#         "DownBlock2D",
#         ),
#         up_block_types=(
#             "UpBlock2D",  # a regular ResNet upsampling block
#             "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
#             "UpBlock2D",
#             "UpBlock2D",
#             "UpBlock2D",
#             "UpBlock2D",
#         ),
#     )
# 
#   # Our forward method now takes the class labels as an additional argument
#   def forward(self, x, t, class_labels):
#     # Shape of x:
#     bs, ch, w, h = x.shape
#     # class conditioning in right shape to add as additional input channels
#     temp_tensor  = torch.tensor([], dtype=torch.int32).to(device)
#     new_element = torch.tensor(24).to(device)
#     #print(class_labels.shape)
#     for instance in class_labels:
#         indices = torch.nonzero(instance).squeeze()
#         if(indices.dim() == 0):
#             indices = indices.unsqueeze(0)
#         for i in range(3-indices.shape[0]):
#             indices = torch.cat((indices, new_element.unsqueeze(0)))
#         temp_tensor = torch.cat((temp_tensor,indices.unsqueeze(0)))
#     
#     tensor1 = temp_tensor.t()[0]
#     tensor2 = temp_tensor.t()[1]
#     tensor3 = temp_tensor.t()[2]
#     class_cond1 = self.class_emb(tensor1) # Map to embedding dinemsion
#     class_cond2 = self.class_emb(tensor2) # Map to embedding dinemsion
#     class_cond3 = self.class_emb(tensor3) # Map to embedding dinemsion
#     
#     class_cond1 = class_cond1.view(bs, class_cond1.shape[1], 1, 1).expand(bs, class_cond1.shape[1], w, h)
#     class_cond2 = class_cond2.view(bs, class_cond2.shape[1], 1, 1).expand(bs, class_cond2.shape[1], w, h)
#     class_cond3 = class_cond3.view(bs, class_cond3.shape[1], 1, 1).expand(bs, class_cond3.shape[1], w, h)
# 
#     # x is shape (bs, 3, 64, 64) and class_cond is now (bs, 4, 64, 64)
# 
#     # Net input is now x and class cond concatenated together along dimension 1
#     net_input = torch.cat((x, class_cond1), 1) # (bs, 7, 64, 64)
#     net_input = torch.cat((net_input, class_cond2), 1) # (bs, 11, 64, 64)
#     net_input = torch.cat((net_input, class_cond3), 1) # (bs, 15, 64, 64)
# 
#     # Feed this to the unet alongside the timestep and return the prediction
#     return self.model(net_input, t).sample # (bs, 3, 64, 64)
# =============================================================================
net = UNet2DModel(
    sample_size = 64,
    in_channels = 3,
    out_channels = 3,
    layers_per_block = 2,
    class_embed_type = None,
    block_out_channels = (128, 128, 256, 256, 512, 512), 
    down_block_types=(
    "DownBlock2D",  # a regular ResNet downsampling block
    "DownBlock2D",
    "DownBlock2D",
    "DownBlock2D",
    "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
    "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

#%%
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
#%%

# How many runs through the data should we do?
epochs = 200

# Our network 
net.class_embedding = nn.Linear(24 ,512)
net = net.to(device)

# Our loss finction
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.AdamW(net.parameters(), lr=1e-4) 

# Keeping a record of the losses for later viewing
losses = []
avg_losses = []
net.train() 
# The training loop
for epoch in range(epochs):
    for img, label in tqdm(train_dataloader):
        
        # Get some data and prepare the corrupted version
        img = img.to(device,dtype=torch.float32)
        label = label.to(device,dtype=torch.float32)
        label = label.squeeze()
        noise = torch.randn_like(img)
        timesteps = torch.randint(0, 999, (img.shape[0],)).long().to(device)
        noisy_img = noise_scheduler.add_noise(img, noise, timesteps)

        # Get the model prediction
        pred = net(sample = noisy_img, timestep = timesteps, class_labels = label) # Note that we pass in the labels y

        # Calculate the loss
        loss = loss_fn(pred.sample, noise) # How close is the output to the noise

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        losses.append(loss.item())

    # Print our the average of the last 100 loss values to get an idea of progress:
    avg_loss = sum(losses)/len(losses)
    avg_losses.append(avg_loss)
    print(f'epoch {epoch}: Average loss = {avg_loss:05f}')
    # torch.save(net.state_dict(), "ckpt/"+"ddpm"+str(epoch)+".pt")
    if(epoch%5==0):
        print("==test.json==")
        test_acc = sample(net, device, test_dataloader, "test",str(epoch))
        print("==test_new.json==")
        new_test_acc = sample(net, device, new_test_dataloader, "new_test",str(epoch))
    
    if test_acc > 0.85 and new_test_acc > 0.85:
        break

# View the loss curve
plt.plot(avg_losses)
plt.title("Average loss")
plt.show()

#%%

















