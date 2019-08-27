import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import os
import pickle
import statistics
import glob

import losses
import models.standard_cnn as standard_cnn
from inception_score import inceptions_score_all_weights

def load_cifar(batch_size):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True,
                transform=trans, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                shuffle=True, num_workers=4)
    return dataloader

def train(cases):
    # case 0
    # standard_cnn + bce loss

    # case 1
    # standard_cnn + hinge_loss

    # case 2
    # standard_cnn_conditional + bce_loss

    # case 3
    # standard_cnn_conditional + hinge_loss

    output_dir = f"cifar_case{cases}"

    batch_size = 64
    device = "cuda:1"

    dataloader = load_cifar(batch_size)

    if cases in [0, 1]:
        enable_conditional = False
    elif cases in [2, 3]:
        enable_conditional = True    
    model_G = standard_cnn.Generator(dataset="cifar", enable_conditional=enable_conditional)
    model_D = standard_cnn.Discriminator(dataset="cifar", enable_conditional=enable_conditional)
    model_G, model_D = model_G.to(device), model_D.to(device)

    param_G = torch.optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.0, 0.9))
    param_D = torch.optim.Adam(model_D.parameters(), lr=0.0002, betas=(0.0, 0.9))

    if cases in [0, 2]:
        gan_loss = losses.DCGANCrossEntropy(batch_size, device)
    elif cases in [1, 3]:
        gan_loss = losses.HingeLoss(batch_size, device)

    n_dis_update = 5

    result = {"d_loss": [], "g_loss": []}
    n = len(dataloader)
    onehot_encoding = torch.eye(10).to(device)

    for epoch in range(321):
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
        torchvision.utils.save_image(fake_img_tensor, f"{output_dir}/epoch_{epoch:03}.png",
                                    nrow=8, padding=2, normalize=True, range=(-1.0, 1.0))

        # 係数保存
        if not os.path.exists(output_dir + "/models"):
            os.mkdir(output_dir+"/models")
        if epoch % 5 == 0:
            torch.save(model_G.state_dict(), f"{output_dir}/models/gen_epoch_{epoch:03}.pytorch")
            torch.save(model_D.state_dict(), f"{output_dir}/models/dis_epoch_{epoch:03}.pytorch")

    # ログ
    with open(output_dir + "/logs.pkl", "wb") as fp:
        pickle.dump(result, fp)
            
def evaluate(cases):
    if cases in [0, 1]:
        enable_conditional = False
        n_classes = 0
    elif cases in [2, 3]:
        enable_conditional = True
        n_classes = 10    

    inceptions_score_all_weights("cifar_case" + str(cases), standard_cnn.Generator,
                                100, 100, dataset="cifar", n_classes=n_classes,
                                enable_conditional=enable_conditional)
    
if __name__ == "__main__":
    for i in range(4):
        train(i)
        evaluate(i)

