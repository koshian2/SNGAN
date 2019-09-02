import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
import glob
import pandas as pd
import os
from scipy.stats import entropy
from tqdm import tqdm

# https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

def inception_score(imgs, cuda=True, batch_size=64, resize=True, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    if cuda:
        inception_model = torch.nn.DataParallel(inception_model)
    inception_model.eval()

    up = nn.UpsamplingBilinear2d(size=(299, 299)).type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

# 10.254

def evaluate_inception_score(image_tensor, verbose=False):
    assert image_tensor.size(1) == 3
    # Range check
    minvalue = torch.min(image_tensor)
    maxvalue = torch.max(image_tensor)
    if minvalue <= -1.01 or minvalue >= -0.98:
        print(f"Image tensor should be [-1, 1] range. Min value = {minvalue}. Do you intended ?")
    if maxvalue >= 1.01 or maxvalue <= 0.98:
        print(f"Image tensor should be [-1, 1] range. Max value = {maxvalue}. Do you intended ?")
    
    # preprocess
    x = image_tensor / 2.0 + 0.5 # [0, 1]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    dataset = IgnoreLabelDataset(torch.utils.data.TensorDataset(x))
    if verbose:
        print("Calculating Inception Score...")
        
    result = inception_score(dataset)

    if verbose:
        print(result)
    return result

def inceptions_score_all_weights(base_dir, generator_class,
                                 generates_mini_batches, batch_size, n_classes=0, *args, **kwargs):
    model_paths = sorted(glob.glob(base_dir + "/models/gen*.pytorch"))

    epochs = []
    inception_scores = []

    print(f"Calculating All Inception Scores...  (# {len(model_paths)})")
    for i, path in enumerate(model_paths):
        model = generator_class(*args, **kwargs)
        model.load_state_dict(torch.load(path))
        model = torch.nn.DataParallel(model.cuda())

        # generate images
        with torch.no_grad():
            model.eval()
            imgs = []
            for _ in range(generates_mini_batches):
                if n_classes == 0: # unconditional
                    x = model(torch.randn(batch_size, 128))
                else:  # conditional
                    label_onehot = torch.eye(n_classes)[torch.randint(0, n_classes, (batch_size,))]
                    x = model(torch.randn(batch_size, 128), label_onehot)
                imgs.append(x)
            imgs = torch.cat(imgs, dim=0).cpu()

        # eval_is
        iscore, _ = evaluate_inception_score(imgs)
        # epoch
        epoch = int(os.path.basename(path).replace("gen_epoch_", "").replace(".pytorch", ""))
        epochs.append(epoch)
        inception_scores.append(iscore)
        print(f"epoch = {epoch}, inception_score = {iscore}    [{i+1}/{len(model_paths)}]")

    df = pd.DataFrame({"epoch": epochs, "inception_score": inception_scores})
    df.to_csv(base_dir+"/inception_score.csv", index=False)

# checking for cifar
def cifar_test():
    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='data/', download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                      std=[0.5, 0.5, 0.5])
                             ])
    )
    loader = torch.utils.data.DataLoader(cifar, batch_size=100)
    batches = []
    for b, _ in loader:
        batches.append(b)
    images = torch.cat(batches, dim=0)
    evaluate_inception_score(images, verbose=True) # should be around 10.25

