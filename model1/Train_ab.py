from Unet import *
from utils import *
import torch
from dataset import CTFatDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import sys

import matplotlib.pyplot as plt

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    prt_green('Cuda version: {}'.format(torch.version.cuda))
    prt_green('Torch version: {}'.format(torch.__version__))
    DEVICE = "cuda"
    prt_cyan('Running on GPU')
else:
    DEVICE = "cpu"
    prt_cyan('Running on CPU')
    
dataFolder = '/Users/vynguyen/Documents/VSCode/Abdominal_segmentation/CT_FAT'

def myDataset(transform= None):
    ds_train= CTFatDataset(images_dir= dataFolder, subset= "train", transform= transform)
    ds_val = CTFatDataset(images_dir= dataFolder, subset= "validation", transform= transform)
    return ds_train, ds_val

transformsImage = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),    
])

def main():
    n_epochs = 150
    batch_size = 4
    lr = 1e-4
    val_interval = 5
    ds_train, ds_val = myDataset(transformsImage)
    
    # load data
    train_loader = DataLoader(ds_train, batch_size = batch_size,
                                shuffle = True, drop_last = True)
    val_loader = DataLoader(ds_val, batch_size = batch_size,
                                shuffle = True, drop_last = True)
    
    plots_path = '/Users/vynguyen/Documents/VSCode/Abdominal_segmentation/plots'
    
    model = UNet(in_channels = 1, out_channels =1)
    model.to(DEVICE, dtype=torch.float)
    dsc_loss = DiceLoss()
    name_loss = dsc_loss._get_name()
    # print(dsc_loss.__name__())
    mplt = myPlot(saveDir= os.path.join(plots_path, '{}_figures'.format(name_loss)),
                  blockShow=False, overwrite_all=True)
    optimizer = optim.Adam(model.parameters(), lr= lr)
    save_model = os.path.join(dataFolder, 'saveModel')
    model_name = "Best_UNet_loss_{}".format(name_loss)
    if not os.path.exists(save_model):
        os.makedirs(save_model)
    
    loss_train = []
    loss_val = []
    step = 0
    best_loss = 1
    best_epoch = 0
    for epoch in range(n_epochs):
        if ((epoch+1)% val_interval) ==0:
            phase = "valid"
            model.eval()
            loader = val_loader
        else: 
            phase = "train"
            model.train()
            loader = train_loader
        
        train_loss = []
        val_loss = []
        
        val_pred = []
        val_true = []
        
        for i, data in enumerate(loader):
            img, mask = data
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase== "train"):
                y_pred = model(img)
                loss = dsc_loss(y_pred, mask)
                
                if phase == "valid":
                    print(img.shape)
                    val_loss.append(loss.item())
                    # detect array from torch-gpu to numpy
                    y_pred_np = y_pred.detach().cpu().numpy()
                    val_pred.append(y_pred_np)
                    img_np = img.detach().cpu().numpy()
                    mask_np = mask.detach().cpu().numpy()
                    val_true.append(mask_np)
                
                if phase == "train":
                    train_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()
        
        # report
        if phase == "train":
            prt_yellow("------------Training result------------")
            prt_cyan("Epoch {}, training loss: {}".format(epoch, train_loss))
        if phase == "valid":
            prt_yellow("------------Validation result------------")
            prt_cyan("Epoch {}, validation loss: {}".format(epoch, val_loss))
            
            # find best model
            min_loss = np.min(val_loss)
            if min_loss < best_loss:
                best_loss = min_loss
                best_epoch = epoch
                prt_pink("The best loss {} at epoch {}".format(best_loss, best_epoch))
                
                with plt.ioff():
                    figure = mplt.show_imgs_list_of_tensor([img_np, mask_np, y_pred_np], noShow= "True")
                    mplt.save_Fig(figure, 'temp{}_epoch{}'.format(name_loss, epoch))
                prt_cyan("************Saving Model************")
                torch.save(model.state_dict(), os.path.join(save_model, model_name))

main()