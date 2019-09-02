import torch
import torchvision
import glob
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import models
import models.standard_cnn, models.cifar_resnet, models.stl_resnet, models.post_act_resnet, models.resnet_size96
import numpy as np
from PIL import Image

def inception_log_graph(base_dir):
    csvs = sorted(glob.glob(base_dir + "_case*/inception_score.csv"))
    datax = []
    datay = []
    cases = []
    model_paths = []
    for c in csvs:
        df = pd.read_csv(c).sort_values(by=["epoch"])
        dir_name = os.path.dirname(c).replace("\\", " / ").split(" / ")[-1]
        regex=re.search('case([0-9]*)', dir_name).group(1)
        cases.append(regex)
        x = df["epoch"].values
        y = df["inception_score"].values
        max_epoch = x[np.argmax(np.array(y))]
        model_paths.append(os.path.dirname(c) + f"/models/gen_epoch_{max_epoch:04}.pytorch")        
        datax.append(x)
        datay.append(y)        
        title=dir_name.replace("_case" + regex, "")

    plt.clf()
    for x, y, c in zip(datax, datay, cases):
        plt.plot(x, y, label="case " + c)
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Inception score")
    plt.savefig("graph/" + title + ".png")

    return model_paths

def sampling_and_interpolation(base_dir):
    experiment_case = base_dir.split("/")[1]
    if experiment_case == "cifar":
        ms = [models.standard_cnn.Generator(enable_conditional=i // 2 == 1) for i in range(4)]
        xs = [torch.randn(50, 128) for i in range(4)]
        cs = [torch.eye(10)[torch.randint(0, 10, (50,))] if i // 2 == 1 else None for i in range(4)]
    elif experiment_case == "cifar_resnet":
        ms = [models.cifar_resnet.Generator(enable_conditional=i // 2 == 1) for i in range(4)]
        xs = [torch.randn(50, 128) for i in range(4)]
        cs = [torch.eye(10)[torch.randint(0, 10, (50,))] if i // 2 == 1 else None for i in range(4)]
    elif experiment_case == "stl":
        ms = [models.standard_cnn.Generator(dataset="stl", enable_conditional=i // 2 == 1) for i in range(4)]
        xs = [torch.randn(50, 128) for i in range(4)]
        cs = [torch.eye(10)[torch.randint(0, 10, (50,))] if i // 2 == 1 else None for i in range(4)]
    elif experiment_case == "stl_resnet":
        ms = [models.stl_resnet.Generator(enable_conditional=i % 2 != 0) for i in range(8)]
        xs = [torch.randn(50, 128) for i in range(8)]
        cs = [torch.eye(10)[torch.randint(0, 10, (50,))] if i % 2 != 0 else None for i in range(8)]
    elif experiment_case == "stl_resnet_postact":
        ms = [models.post_act_resnet.Generator(latent_dims=3, n_classes_g=10 if i % 2 != 0 else 0) for i in range(10)]
        xs = [torch.randn(50, 128) for i in range(10)]
        cs = [torch.eye(10)[torch.randint(0, 10, (50,))] if i % 2 != 0 else None for i in range(10)]
    elif experiment_case == "stl_resnet_postact2":
        ms = [models.post_act_resnet.Generator(latent_dims=3, n_classes_g=10) for i in range(6)]
        xs = [torch.randn(50, 128) for i in range(6)]
        cs = [torch.eye(10)[torch.randint(0, 10, (50,))] for i in range(6)]
    elif experiment_case == "stl_resnet_dchange":
        ms = [models.stl_resnet.Generator(enable_conditional=True) for i in range(10)]
        xs = [torch.randn(50, 128) for i in range(4)]
        cs = [torch.eye(10)[torch.randint(0, 10, (50,))] for i in range(4)]
    elif experiment_case == "anime":
        ms = [models.resnet_size96.Generator(n_classes_g=i) for i in [0, 176]]
        xs = [torch.randn(32, 128) for i in range(2)]
        cs = [None, torch.eye(176)[torch.randint(0, 176, (32,))]]
    elif experiment_case == "flower":
        ms = [models.resnet_size96.Generator(n_classes_g=i) for i in [0, 102]]
        xs = [torch.randn(32, 128) for i in range(2)]
        cs = [None, torch.eye(102)[torch.randint(0, 102, (32,))]]



    # animefaceの場合、Inception Scoreが意味がないので（ドメイン相違）最後の係数を使う
    # flowerはanimefaceよりInceptionが使えないわけではないが、ISがあまり当てにならない
    if experiment_case == "anime":
        paths = [f"{base_dir}_case{i}/models/gen_epoch_1100.pytorch" for i in range(2)]
    elif experiment_case == "flower":
        paths = [f"{base_dir}_case{i}/models/gen_epoch_1900.pytorch" for i in range(2)]
    else:
        paths = inception_log_graph(base_dir)
        



    for m, x, c, p in zip(ms, xs, cs, paths):
        m.load_state_dict(torch.load(p))

        # sampling
        y = m(x, c) if c is not None else m(x)
        n_row = int(np.rint(np.sqrt(len(x) // 2)))
        sam_img = torchvision.utils.make_grid(y, normalize=True, range=(-1.0, 1.0), nrow=n_row)
        # interpolation
        k = torch.arange(n_row, dtype=torch.float) / (n_row - 1)
        k = k.view(1, n_row, 1)
        n_col = len(x) // n_row
        x1 = x[:n_col].view(n_col, 1, -1)
        x2 = x[n_col:(2 * n_col)].view(n_col, 1, -1)        
        interx = k * x1 + (1 - k) * x2
        interx = interx.view(-1, interx.size(2))
        if c is not None:
            interc = c[:n_col].view(n_col, 1, -1).expand(n_col, n_row, -1).contiguous().view(-1, c.size(1))
            y = m(interx, interc)
        else:
            y = m(interx)
        # "interpolation_" + os.path.dirname(p).replace("\\", "/").split("/")[-2]+".png"
        inter_img = torchvision.utils.make_grid(y, normalize=True, range=(-1.0, 1.0), nrow=n_row)

        # padding
        pad = torch.zeros(sam_img.size(0), sam_img.size(1), 20)

        # concat
        img = torch.cat([sam_img, pad, inter_img], dim=2).permute([1, 2, 0]).detach().numpy()
        img = (img * 255.0).astype(np.uint8)
        with Image.fromarray(img) as img:
            img.save("sampling_interpolation/"+ os.path.dirname(p).replace("\\", "/").split("/")[-2]+".png")        


if __name__ == "__main__":
    sampling_and_interpolation("results/flower")
    

