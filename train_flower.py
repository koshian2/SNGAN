import torch
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import os
import pickle
import statistics
import glob
import shutil

import losses
import models.resnet_size96 as resnet96
import models.resnet_size96_light as resnet96_light
from inception_score import inceptions_score_all_weights

def load_animeface(batch_size):
    trans = transforms.Compose([
        transforms.Resize(size=(96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(root="./data/flower", transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

def train(cases):
    # Flower (96x96)
    # case = 0 : unconditional
    # case = 1 : conditional

    output_dir = f"flower_case{cases}"

    batch_size = 64
    device = "cuda"

    enable_conditional = (cases == 1)

    dataloader = load_animeface(batch_size)

    model_G = resnet96.Generator(n_classes_g=102 if enable_conditional else 0)
    model_D = resnet96_light.DiscriminatorLight(n_classes_d=102 if enable_conditional else 0) # Dを軽くすると53s/ep -> 35s/ep
    # model_D = resnet96.Discriminator(n_classes_d=102 if enable_conditional else 0)
    model_G, model_D = model_G.to(device), model_D.to(device)

    param_G = torch.optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.9))
    param_D = torch.optim.Adam(model_D.parameters(), lr=0.0002, betas=(0.5, 0.9))

    gan_loss = losses.HingeLoss(batch_size, device)

    n_dis_update = 5
    n_epoch = 1901

    result = {"d_loss": [], "g_loss": []}
    n = len(dataloader)
    onehot_encoding = torch.eye(102).to(device)

    for epoch in range(n_epoch):
        log_loss_D, log_loss_G = [], []

        for i, (real_img, labels) in tqdm(enumerate(dataloader), total=n):
            batch_len = len(real_img)
            real_img = real_img.to(device)
            if enable_conditional:
                label_onehots = onehot_encoding[labels.to(device)] # conditional
            else:
                label_onehots = None # non conditional
            
            # train G
            if i % n_dis_update == 0:
                param_G.zero_grad()
                param_D.zero_grad()

                rand_X = torch.randn(batch_len, 128).to(device)
                if enable_conditional:
                    fake_img = model_G(rand_X, label_onehots)
                else:
                    fake_img = model_G(rand_X)
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
        if epoch % 4 == 0:
            torchvision.utils.save_image(fake_img_tensor[:25], f"{output_dir}/epoch_{epoch:03}.png",
                                        nrow=5, padding=5, normalize=True, range=(-1.0, 1.0))

        # 係数保存
        if not os.path.exists(output_dir + "/models"):
            os.mkdir(output_dir+"/models")
        if epoch % 20 == 0:
            torch.save(model_G.state_dict(), f"{output_dir}/models/gen_epoch_{epoch:04}.pytorch")
            torch.save(model_D.state_dict(), f"{output_dir}/models/dis_epoch_{epoch:04}.pytorch")

    # ログ
    with open(output_dir + "/logs.pkl", "wb") as fp:
        pickle.dump(result, fp)
            
def evaluate(cases):
    if cases == 0:
        n_classes = 0
    elif cases == 1:
        n_classes = 102    

    inceptions_score_all_weights("flower_case" + str(cases), resnet96.Generator,
                                100, 100, n_classes=n_classes, n_classes_g=n_classes)
    
if __name__ == "__main__":
    for i in range(2):
        train(i)
        evaluate(i)
