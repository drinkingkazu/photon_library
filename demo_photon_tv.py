import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import nn

from photon_library import PhotonLibrary


class SineLayer(nn.Module):
    # Copied from https://github.com/vsitzmann/siren
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    # Copied from https://github.com/vsitzmann/siren
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


def tv_loss(data, v_shape, tv_dim):
    x = torch.reshape(data, v_shape)

    # Need at least three slices to compute tv loss in x direction
    if tv_dim == 3 and v_shape[0] > 2:
        x_var = torch.mean(torch.abs(x[1:, :, :] - x[:-1, :, :]))
    else:
        x_var = 0

    z_var = torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]))
    y_var = torch.mean(torch.abs(x[:, 1:, :] - x[:, :-1, :]))

    return x_var + y_var + z_var


parser = argparse.ArgumentParser()
parser.add_argument('--start_x', type=int, default=1,
                    help='Starting slice in x axis')
parser.add_argument('--end_x', type=int, default=10,
                    help='Ending slice in x axis')
parser.add_argument('--total_steps', type=int, default=100,
                    help='Number of optimization iterations')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--tv_weight', type=float, default=0.01,
                    help='Weight for Total Variation Loss')
parser.add_argument('--tv_dim', type=int, default=3,
                    help='Number of dimensions for TV loss')
parser.add_argument('--output_dir', type=str, default="./result",
                    help='Weight for Total Variation Loss')
parser.add_argument('--render_scale', type=float, default=255.0,
                    help='Weight for Total Variation Loss')
parser.add_argument('--print_steps', type=int, default=50,
                    help='Number of steps to print loss')
parser.add_argument('--train', action="store_true", default=False,
                    help="Flag to train. If not passed the script will try to find a pre-trained model for running inference")
parser.add_argument('--add_err', action="store_true", default=False,
                    help="Flag to add error to the visualization.")
params = parser.parse_args()

print('Load data ...')
plib = PhotonLibrary()
data_all = plib.numpy()

for slice_size in range(params.end_x, params.start_x, -5):

    print('A Siren model for the first {}-th slices for {} iterations'.format(slice_size, params.total_steps))
    
    output_path = params.output_dir + '/photon_' + str(slice_size)
    weight_path = params.output_dir + '/photon_' + str(slice_size) + '_weights.pth' 

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data = data_all[:slice_size, :, :, :]
    data_shape = data.shape[0:-1]

    data = np.expand_dims(np.reshape(data, (-1, data.shape[-1])), axis=0)
    data = np.expand_dims(np.sum(data, -1), axis=-1)
    # data = -np.log(data + 1.e-7)
    data = (data - np.amin(data)) / (np.amax(data) - np.amin(data)) - 0.5
    data = torch.from_numpy(data.astype(np.float32)).cuda()
    data.requires_grad = False

    x = np.linspace(0, data_shape[0] - 1, data_shape[0])
    y = np.linspace(0, data_shape[1] - 1, data_shape[1])
    z = np.linspace(0, data_shape[2] - 1, data_shape[2])

    coordx, coordy, coordz = np.meshgrid(x, y, z)
    coord = np.reshape(np.stack([coordx, coordy, coordz], -1), (-1, 3))
    coord_real = np.expand_dims(plib._min + (plib._max - plib._min) / plib.shape * (coord + 0.5), axis=0)
    coord_real = torch.from_numpy(coord_real.astype(np.float32)).cuda()
    coord_real.requires_grad = False

    img_siren = Siren(in_features=3, out_features=1, hidden_features=256, 
                      hidden_layers=3, outermost_linear=True)
    img_siren.cuda()

    if params.train:
        schedule_dict = {
            'fixed': lambda x: 1,
            'lineardecay': lambda x: 1.0 - x/params.total_steps,
        }

        schedule_func = schedule_dict['lineardecay']
        optim = torch.optim.Adam(lr=params.lr, params=img_siren.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, schedule_func)


        for step in range(params.total_steps):
            model_output, coords = img_siren(coord_real)    

            loss_tv = tv_loss(model_output, data_shape, params.tv_dim)

            loss_rec = ((model_output - data)**2).mean()   

            loss = params.tv_weight * loss_tv + loss_rec

            if step % params.print_steps == 0:
                print('rec loss {:.5f}, tv loss {:.5f}, total loss {:.5f}'.format(loss_rec, loss_tv, loss))

            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
    else:
        checkpoint = torch.load(weight_path)
        img_siren.load_state_dict(checkpoint)
        img_siren.eval()

    model_output, coords = img_siren(coord_real)


    ground_truth_video = np.reshape(
      data.cpu().detach().numpy(), 
      (data_shape[0], data_shape[1], data_shape[2], 1)
    )

    predict_video = np.reshape(
        model_output.cpu().detach().numpy(), 
        (data_shape[0], data_shape[1], data_shape[2], 1)
    )

    diff = np.sum(abs(ground_truth_video - predict_video), axis=(1,2,3))

    ground_truth_video = np.uint8((ground_truth_video * 1.0 + 0.5) * params.render_scale)
    predict_video = np.uint8((predict_video * 1.0 + 0.5) * params.render_scale)

    render_video = np.concatenate((ground_truth_video, predict_video), axis=1)

    for step in range(render_video.shape[0]):
        im_name = os.path.join(output_path, '{:05d}.png'.format(step))
        im_render = Image.fromarray(np.squeeze(render_video[step], -1) , 'L').convert('RGB')

        if params.add_err:
            draw = ImageDraw.Draw(im_render)
            font = ImageFont.truetype("./font/Arsenal-Regular.otf", 16)
            draw.text((0, 0), "{:.2f}".format(diff[step]), (255,0,0), font=font)

        im_render.save(im_name, 'png')

    torch.save(img_siren.state_dict(), weight_path)