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
    ## ResNet version stl-10 (post-act, D=small non-resnet)

    # case 0
    # n_dis = 5, beta2 = 0.9, non-conditional
    # case 1
    # n_dis = 5, beta2 = 0.9, conditional
    # case 2
    # n_dis = 1, beta2 = 0.9, non-conditional
    # case 3
    # n_dis = 1, beta2 = 0.9, conditional
    # case 4
    # n_dis = 1, beta2 = 0.999, non-conditional
    # case 5
    # n_dis = 1, beta2 = 0.999, conditional
    # case 6
    # n_dis = 1, beta2 = 0.999, non-conditional, leaky_relu_slope = 0.2 (others=0.1)
    # case 7
    # n_dis = 1, beta2 = 0.999, conditional, leaky_relu_slope = 0.2
    # case 8
    # n_dis = 1, beta2 = 0.999, non-conditional, leaky_relu_slope = 0.2, lr_d = 0.001 (others=0.0002)
    # case 9
    # n_dis = 1, beta2 = 0.999, conditional, leaky_relu_slope = 0.2, lr_d = 0.001

    output_dir = f"stl_resnet_postact_case{cases}"

    batch_size = 64
    device = "cuda"

    dataloader = load_stl(batch_size)

    n_classes = 10 if (cases % 2 != 0) else 0 # Conditional / non-Conditional
    n_dis_update = 5 if cases <= 1 else 1
    beta2 = 0.9 if cases <= 3 else 0.999
    lrelu_slope = 0.1 if cases <= 5 else 0.2
    lr_d = 0.0002 if cases <= 7 else 0.001

    n_epoch = 1301 if n_dis_update == 5 else 261
    
    model_G = post_act_resnet.Generator(latent_dims=3, n_classes_g=n_classes)
    model_D = post_act_resnet.Discriminator(latent_dims=3, n_classes=n_classes, lrelu_slope=lrelu_slope)
    model_G, model_D = model_G.to(device), model_D.to(device)

    param_G = torch.optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, beta2))
    param_D = torch.optim.Adam(model_D.parameters(), lr=lr_d, betas=(0.5, beta2))

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
            if n_classes != 0:
                label_onehots = onehot_encoding[labels.to(device)] # conditional
            else:
                label_onehots = None # non conditional
            
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
    if cases % 2 == 0:
        enable_conditional = False
        n_classes = 0
    else:
        enable_conditional = True
        n_classes = 10    

    inceptions_score_all_weights("stl_resnet_postact_case" + str(cases), post_act_resnet.Generator,
                                100, 100, n_classes=n_classes, latent_dims=3, n_classes_g=n_classes)
    
if __name__ == "__main__":
    for i in range(10):
        train(i)
        evaluate(i)
