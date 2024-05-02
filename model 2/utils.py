import numpy as np
# from medpy.filter.binary import largest_connected_component
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import sys, os
import sklearn
from sklearn.preprocessing import maxabs_scale, minmax_scale
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from scipy.spatial.distance import dice


# def dsc(y_pred, y_true, lcc=True):
#     if lcc and np.any(y_pred):
#         y_pred = np.round(y_pred).astype(int)
#         y_true = np.round(y_true).astype(int)
#         # y_pred = largest_connected_component(y_pred)
#     return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

def prt_red(str): print("\033[91m {}\033[00m". format(str))

def prt_green(str): print("\033[92m {}\033[00m". format(str))

def prt_yellow(str): print("\033[93m {}\033[00m". format(str))

def prt_lightpurple(str): print("\033[94m {}\033[00m". format(str))

def prt_pink(str): print("\033[95m {}\033[00m". format(str))

def prt_cyan(str): print("\033[96m {}\033[00m". format(str))


def resize_sample(image, size=256):
    image_shape = image.shape
    out_shape = (size, size)    
    res = resize(
        image,
        output_shape=out_shape,
        order=0,
        mode="constant",
        cval=0,
        anti_aliasing= True,
    )
    return res

def MinMaxScaler(images, setZerostoNan = False):
    if setZerostoNan:
        images= np.where(images==0, np.nan, images)
    res = (images - np.nanmin(images))/(np.nanmax(images)-np.nanmin(images))
    res = np.where(np.isnan(res), 0, res)
    return res

# def MaxAbsScaler(images, setZerostoNan = False)
#     if setZerostoNan:
#         images= np.where(images==0, np.nan, images)
#     res = maxabs_scale(images)
#     res = np.where(np.isnan(res), 0, res)
#     return res

def preprocess(image, func= MinMaxScaler):
    img = image + np.abs(np.min(image))
    new_img = np.zeros(image.shape)
    for i in range(image.shape[-1]):
        new_img[...,i] = func(image[...,i])
    return new_img

def compute_dice(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    dsc = 1- dice(y_true, y_pred)
    return dsc

def resize_image(image, size = 256):
    image_shape = image.shape
    out_shape = (size, size)
    out_img = resize(image, output_shape= out_shape, order= 0, mode= 'constant',
                    anti_aliasing= True)
    return out_img

class HiddenPrints:
    def __init__(self):
        self._origin_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._origin_stdout
        
csfont = {'fontname': 'Times New Roman',
            'fontweight': 'bold'}
        
class myPlot():
    def __init__(self,
                saveDir = '/Users/vynguyen/Documents/VSCode/Abdominal_segmentation/model 2/plots_m2',
                figsize = (18,12), origin_top_left = False, blockShow = True,
                overwrite_all = False
                ):
        self.saveDir = saveDir
        self.blockShow = blockShow
        self.figsize = figsize
        self.overwrite_all = overwrite_all 
        self.origin_top_left = origin_top_left
        self.hspace = 0.1
        self.wspace = 0.3
        self.titlepad = 0.8
        self.origin = None
        self.create_save_folder()
        
        if origin_top_left:
            self.origin = 'lower'
        else:
            self.origin = 'upper'
    
    def create_save_folder(self):
        if not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)
    
    def save_Fig(self, fig, fig_name = 'Figure', overwrite = False):
        if self.overwrite_all: overwrite= True
        fig_filename = os.path.join(self.saveDir, '{}.png'.format(fig_name))
        
        if os.path.exists(fig_filename):
            if not overwrite:
                print('Exist image: {}'.format(fig_name))
                return fig_name
        print('Saving image to {}'.format(fig_name))
        fig.savefig(fig_filename, bbox_inches= 'tight', transparent= True,
                    pad_inches = 0)
        return fig_filename
    
    def show_imgs(self, list_imgs, imageNames = None,
                main_title = None, figsize = None, colormap = 'turbo',
                vmax = None, vmin = None):
        
        # image can be list or np array with last 
        if isinstance(list_imgs, list):
            img_array = np.squeeze(np.stack(list_imgs, axis= 1))
        if isinstance(list_imgs, np.ndarray):
            img_array = list_imgs
            
        if figsize is None:
            figsize = self.figsize
        n_img = img_array.shape[-1]
        
        if len(img_array.shape)== 2:
            n_img = 1
            fig, ax = plt.subplots(1, n_img, figsize= figsize)
            ax.imshow(list_imgs, cmap= colormap, vmax= vmax, vmin= vmin)
            ax.axis('off')
            if main_title:
                fig.suptitle(main_title)
            plt.show(block = self.blockShow)
            return fig
        
        if n_img > 4:
            n_col = 4
            n_row = n_img // n_col+1
            fig, ax = plt.subplots(n_row, n_col, figsize= figsize)
        else:
            fig, ax = plt.subplots(1, n_img, figsize= figsize)
        if imageNames is None:
            title = ['Fig {}'.format(i) for i in range(n_img)]
        else:
            title = imageNames
        
        ax = ax.flatten()
        for img in range(n_img):
            ax[img].imshow(img_array[...,img], cmap= colormap,
                           vmax= vmax, vmin= vmin)
            ax[img].axis('off')
            ax[img].set_title(title[img], **csfont)
            
        if main_title:
            fig.suptitle(main_title, y= self.titlepad, **csfont)
        plt.subplots_adjust(wspace= self.wspace, hspace= self.hspace)
        plt.show(block = self.blockShow)
        
        return fig

    def show_imgs_tensor(self, tensor_img, imageNames = None,
                main_title = None, figsize = None, colormap = 'turbo',
                vmax = None, vmin = None):
            
        if figsize is None:
            figsize = self.figsize
        n_img = tensor_img.shape[0]
        
        if n_img == 1:
            fig, ax = plt.subplots(1, n_img, figsize= figsize)
            ax.imshow(tensor_img[0,0,:,:], cmap= colormap, vmax= vmax, vmin= vmin)
            ax.axis('off')
            if main_title:
                fig.suptitle(main_title)
            plt.show(block = self.blockShow)
            return fig
        
        if n_img > 2:
            n_col = 2
            n_row = n_img // n_col+1
            fig, ax = plt.subplots(n_row, n_col, figsize= figsize)
        else:
            fig, ax = plt.subplots(1, n_img, figsize= figsize)
            
        if imageNames is None:
            title = ['Fig {}'.format(i) for i in range(n_img)]
        else:
            title = imageNames
        
        ax = ax.flatten()
        for img in range(n_img):
            ax[img].imshow(tensor_img[img, 0,...], cmap= colormap,
                           vmax= vmax, vmin= vmin)
            ax[img].axis('off')
            ax[img].set_title(title[img], **csfont)
            
        if main_title:
            fig.suptitle(main_title, y= self.titlepad, **csfont)
        plt.subplots_adjust(wspace= self.wspace, hspace= self.hspace)
        plt.show(block = self.blockShow)
        
        return fig

    def show_imgs_list_of_tensor(self, list_tensor_imgs, imageNames = None,
                main_title = None, figsize = None, colormap = 'turbo',
                vmax = None, vmin = None, noShow= False):
            
        if figsize is None:
            figsize = self.figsize
        if not isinstance(list_tensor_imgs, list):
            prt_red('INPUT SHOULD BE A LIST FOR THIS FUNCTION')
            return None
        n_row = len(list_tensor_imgs)
        n_img = list_tensor_imgs[0].shape[0]
        
        if n_img > 4:
            prt_red('A TENSOR SHOULD HAVE LESS THAN 4 Images')
            return None
        
        n_col = n_img
        fig, ax = plt.subplots(n_row, n_col, figsize= figsize)

        if imageNames is None:
            title = ['Fig {}'.format(i) for i in range(n_img)]
        else:
            title = imageNames
        
        ax = ax.flatten()
        for r in range(n_row):
            tensor_imgs = list_tensor_imgs[r]
            for img in range(n_img):            
                ax[r*n_img + img].imshow(tensor_imgs[img, 0,...], cmap= colormap,
                            vmax= vmax, vmin= vmin)
                ax[r*n_img + img].axis('off')
                ax[r*n_img + img].set_title(title[img], **csfont)
            
        if main_title:
            fig.suptitle(main_title, y= self.titlepad, **csfont)
        plt.subplots_adjust(wspace= self.wspace, hspace= self.hspace)
        plt.show(block = self.blockShow)
        
        if noShow:
            plt.close()
        else:
            plt.show(block= self.blockShow)
        
        return fig
    

    
