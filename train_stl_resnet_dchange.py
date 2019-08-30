import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import os
import pickle
import statistics
import glob

import losses
import models.post_act_resnet as post_act_resnet
import models.stl_resnet_light as stl_resnet_light
import models.stl_resnet as stl_resnet
from inception_score import inceptions_score_all_weights

def load_stl(batch_size):
    # first, store as tensor
    trans = transforms.Compose([
        transforms.Resize(size=(48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # train + test (# 13000)
    dataset = torchvision.datasets.STL10(root="./data", split="train", transform=trans, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    imgs, labels = [], []
    for x, y in dataloader:
        imgs.append(x)
        labels.append(y)
    dataset = torchvision.datasets.STL10(root="./data", split="test", transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    for x, y in dataloader:
        imgs.append(x)
        labels.append(y)
    # as tensor
    all_imgs = torch.cat(imgs, dim=0)
    all_labels = torch.cat(labels, dim=0)
    # as dataset
    dataset = torch.utils.data.TensorDataset(all_imgs, all_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

def train(cases):
    ## ResNet version stl-10
    # Gは論文流用して、Dは自作にLeakyReLUにする    

    # case 0
    # beta2 = 0.9, D = post_act_resnet.Discriminator
    # case 1
    # beta2 = 0.999, D = post_act_resnet.Discriminator
    # case 2
    # beta2 = 0.9, D = post_act_resnet.Discriminator
    # case 3
    # beta2 = 0.999, D = post_act_resnet.Discriminator


    beta2 = 0.9 if cases % 2 == 0 else 0.999

    output_dir = f"stl_resnet_dchange_case{cases}"

    batch_size = 64
    device = "cuda"

    dataloader = load_stl(batch_size)

    n_dis_update = 5
    n_epoch = 1301
    
    model_G = stl_resnet.Generator(enable_conditional=True)
    if cases // 2 == 0:
        model_D = post_act_resnet.Discriminator(latent_dims=3, n_classes=10, lrelu_slope=0.2)
    elif cases // 2 == 1:
        model_D = stl_resnet_light.DiscriminatorStrided(enable_conditional=True)    
    model_G, model_D = model_G.to(device), model_D.to(device)

    param_G = torch.optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, beta2))
    param_D = torch.optim.Adam(model_D.parameters(), lr=0.0002, betas=(0.5, beta2))

    gan_loss = losses.HingeLoss(batch_size, device)

    result = {"d_loss": [], "g_loss": []}
    n = len(dataloader)
    onehot_encoding = torch.eye(10).to(device)

    for epoch in range(n_epoch):
        log_loss_D, log_loss_G = [], []

        for i, (real_img, labels) in tqdm(enumerate(dataloader), total=n):
            batch_len = len(real_img)
            if batch_len != batch_size: continue

            real_img = real_img.to(device)
            label_onehots = onehot_encoding[labels.to(device)] # conditional
            
            # train G
            if i % n_dis_update == 0:
                param_G.zero_grad()
                param_D.zero_grad()

                rand_X = torch.randn(batch_len, 128).to(device)
                fake_img = model_G(rand_X, label_onehots)
                fake_img_tensor = fake_img.detach()
                fake_img_onehots = label_onehots.detach() if label_onehots is not None else None
                g_out = model_D(fake_img, label_onehots)
                loss = gan_loss(g_out, "gen")
                log_loss_G.append(loss.item())
                # backprop
                loss.backward()
                param_G.step()

            # train D
            param_G.zero_grad()
            param_D.zero_grad()
            # train real
            d_out_real = model_D(real_img, label_onehots)
            loss_real = gan_loss(d_out_real, "dis_real")
            # train fake
            d_out_fake = model_D(fake_img_tensor, fake_img_onehots)
            loss_fake = gan_loss(d_out_fake, "dis_fake")
            loss = loss_real + loss_fake
            log_loss_D.append(loss.item())
            # backprop
            loss.backward()
            param_D.step()

        # ログ
        result["d_loss"].append(statistics.mean(log_loss_D))
        result["g_loss"].append(statistics.mean(log_loss_G))
        print(f"epoch = {epoch}, g_loss = {result['g_loss'][-1]}, d_loss = {result['d_loss'][-1]}")        
            
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if epoch % n_dis_update == 0:
            torchvision.utils.save_image(fake_img_tensor, f"{output_dir}/epoch_{epoch:03}.png",
                                        nrow=8, padding=2, normalize=True, range=(-1.0, 1.0))

        # 係数保存
        if not os.path.exists(output_dir + "/models"):
            os.mkdir(output_dir+"/models")
        if epoch % (5 * n_dis_update) == 0:            
            torch.save(model_G.state_dict(), f"{output_dir}/models/gen_epoch_{epoch:04}.pytorch")
            torch.save(model_D.state_dict(), f"{output_dir}/models/dis_epoch_{epoch:04}.pytorch")

    # ログ
    with open(output_dir + "/logs.pkl", "wb") as fp:
        pickle.dump(result, fp)
            
def evaluate(cases):
    inceptions_score_all_weights("stl_resnet_dchange_case" + str(cases), stl_resnet.Generator,
                                100, 100, n_classes=10,
                                enable_conditional=True)
    
if __name__ == "__main__":
    train(3)
