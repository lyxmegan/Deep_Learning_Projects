import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def to_numpy(location):
    return np.float32([location[0], location[2]])

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def to_numpy(location):
    return np.float32([location[0], location[2]])

def extract_peak(heatmap, max_pool_ks=7, min_score=-1, max_det=1):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    
    output = torch.nn.functional.max_pool2d(heatmap[None,None], kernel_size=max_pool_ks, padding=max_pool_ks//2, stride=1).view(-1)
    hm = heatmap.view(-1)

    # get positions where peaks were found
    peaks = torch.eq(hm, output)
    not_peaks = (~peaks).float()
    peaks = peaks.float()

    max_locations = torch.zeros(hm.shape)
    max_locations = peaks * hm - 255 * not_peaks

    # get top max_det values with their flattened index
    if max_locations.shape[0] > max_det:
        values, indices = torch.topk(max_locations, max_det)
    else:
        values, indices = torch.topk(max_locations, max_locations.shape[0])

    return [(int(i) % heatmap.shape[1], int(i) // heatmap.shape[1])
    for v, i in zip(values.cpu(), indices.cpu()) if v > min_score]
 
#use the model from hw4
class Detector(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            # self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            # self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.skip(x))


    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))
            
    def __init__(self, layers=[16, 16, 32, 32], n_input_channels = 3, n_output_channels=1, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
        self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

        c = n_input_channels
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]

        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d'%i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d'%i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        
        return self.classifier(z)


    def detect(self, image, player_info):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: List of detections [(class_id, score, cx, cy), ...],
                    return no more than 100 detections per image
           Hint: Use extract_peak here
        """
        global last_position
        self.eval()
        # get heatmap from model
        heatmap = self.forward(image)[0][0]
        # extract peak 
        peaks = extract_peak(heatmap)
        # if the puck was detected on screen, get x and y values
        if (len(peaks) > 0):
            detect_puck = True
            x = peaks[0][0]
            y = peaks[0][1]
            if(x > 200):
                last_position = 1
            else:
                last_position = -1
        else:
            detect_puck = False
            x = 0 
            y = 0

        if detect_puck:
            proj = np.array(player_info.camera.projection).T
            view = np.array(player_info.camera.view).T
            world_postion= to_numpy(to_world(x, y, proj, view, height=0, W=400, H=300))
        
        else:
            world_postion = None

        return detect_puck, [x , y], world_postion, last_position
    
last_position= -1

def to_world(x, y, proj, view, height=0, W=128, H=96):
    screen_width = 400
    screen_height = 300
    x, y, W, H = x, y, screen_width, screen_height
    pv_inv = np.linalg.pinv(proj @ view)
    xy, d = pv_inv.dot([float(x) / (W / 2) - 1, 1 - float(y) / (H / 2), 0, 1]), pv_inv[:, 2]
    x0, x1 = xy[:-1] / xy[-1], (xy+d)[:-1] / (xy+d)[-1]
    t = (height-x0[1]) / (x1[1] - x0[1])
    return t * x1 + (1-t) * x0

def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r



# if __name__ == '__main__':
#     """
#     Shows detections of your detector
#     """
#     from .utils import DetectionSuperTuxDataset
#     dataset = DetectionSuperTuxDataset('drive_data/train/')
#     import torchvision.transforms.functional as TF
#     from pylab import show, subplots
#     import matplotlib.patches as patches
#     import pystk
    
#     state = pystk.WorldState()
#     player_info = state.players[0]
#     fig, axs = subplots(3, 4)
#     model = load_model()
#     for i, ax in enumerate(axs.flat):
#         im, puck = dataset[i]
#         ax.imshow(TF.to_pil_image(im), interpolation=None)
#         p, cx, cy, l, w = model.detect(im, player_info)
#         print('x, y', cx, cy )
#         ax.add_patch(patches.Circle((cx, cy), radius=max(2  / 2, 0.1), color='rgb'[0]))
#         ax.axis('off')
#     show()