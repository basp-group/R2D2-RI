from model.network.unet_model import UnetModel
from model.network.unetWDSR_model import UnetModel as UnetWDSRModel

import torch
import glob
import gc
import os

def get_DNNs(num_iter: int, 
             ckpt_path):
    """Check if all DNNs are available in the specified path and return a dictionary containing the paths to all DNNs.

    Parameters
    ----------
    num_iter : int
        Number of iterations in the R2D2 series.
        
    ckpt_path : str
        Path to the directory containing checkpoints of the series.

    Returns
    -------
    dict
        Dictionary containing the paths to all DNNs.

    Raises
    ------
    ValueError
        If checkpoint for any iteration is not found or if there is a conflict in checkpoint files.
    """
    dnns_dict = {}
    for i in range(num_iter):
        dnn = glob.glob(os.path.join(ckpt_path, f'*N{i+1}.ckpt'))
        if len(dnn) == 0:
            raise ValueError(f'Checkpoint for N{i+1} not found')
        elif len(dnn) >1:
            raise ValueError(f'Checkpoint conflict for N{i+1}')
        model_loaded = torch.load(dnn[0], map_location=torch.device('cpu'), weights_only=False)
        dnns_dict.update({f'N{i+1}': model_loaded['state_dict']})
    print('All DNNs found.')
    return dnns_dict

def load_state_dict_from_state_dict(net, state_dict):
    state_dict = {'.'.join(f.split('.')[1:]): state_dict[f] for f in state_dict}
    net.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    return net

def load_net(net, cur_iter, layers, dnn_dict):
    state_dict = dnn_dict[f'N{cur_iter}']#.to(args.device)
    if layers == 1:
        net = load_state_dict_from_state_dict(net, state_dict)
    elif layers > 1:
        for i in range(layers):
            net_tmp = {f.split(f'unet{i+1}.')[1]: state_dict[f] for f in state_dict if f.startswith(f'unet{i+1}.')}
            net[i].load_state_dict(net_tmp)
            del net_tmp
        del state_dict
    gc.collect()
    return net

def create_net(args, device):
    """Create the network from the specified checkpoint file.

    Parameters
    ----------
    args : _ArgumentParser
        Arguments parsed from command line and processed.
    device : torch.device
        Device to be used for the network.

    Returns
    -------
    torch.nn.Module
        Network created from the specified checkpoint file.
    """
    load_dict = False
    if args.resume or 'test' in args.mode:
        assert args.checkpoint is not None, 'Checkpoint must be provided.'
        state_dict = torch.load(args.checkpoint, map_location=device)['state_dict']
        load_dict = True
    match args.architecture:
        case 'unet':
            net = UnetModel
            split_key = 'unet.'
        case 'uwdsr':
            net = UnetWDSRModel
            split_key = 'model.'
    if args.layers == 1:
        unet = net(in_chans=2,
                    out_chans=1,
                    chans=args.num_chans,
                    num_pool_layers=args.num_pools,
                    drop_prob=0.).to(device)
        if load_dict:
            net_tmp = {'.'.join(f.split('.')[1:]): state_dict[f] for f in state_dict}
            unet.load_state_dict(net_tmp)
    elif args.layers > 1:
        unets = []
        for i in range(args.layers):
            unet = net(in_chans=2,
                        out_chans=1,
                        chans=args.num_chans,
                        num_pool_layers=args.num_pools,
                        drop_prob=0.).to(device)
            if load_dict:
                net_tmp = {f.split(f'{i+1}.')[1:]: state_dict[f] for f in state_dict}
                unet.load_state_dict(net_tmp)

            unets.append(unet)
        unet = unets
    if load_dict:
        del state_dict, net_tmp
        gc.collect()
    return unet

def create_net_imaging(layers: int = 1,
                         num_chans: int = 64,
                         num_pools: int = 4, 
                         architecture: str = 'unet',
                         device: torch.device = torch.device('cpu')):
    """Create the network from the specified checkpoint file.

    Parameters
    ----------
    device : torch.device
        Device to be used for the network.

    Returns
    -------
    torch.nn.Module
        Network created from the specified checkpoint file.
    """
    match architecture:
        case 'unet':
            net = UnetModel
        case 'uwdsr':
            net = UnetWDSRModel
    if layers == 1:
        unet = net(in_chans=2,
                    out_chans=1,
                    chans=num_chans,
                    num_pool_layers=num_pools,
                    drop_prob=0.).to(device)
    elif layers > 1:
        unets = []
        for _ in range(layers):
            unet = net(in_chans=2,
                        out_chans=1,
                        chans=num_chans,
                        num_pool_layers=num_pools,
                        drop_prob=0.).to(device)
            unets.append(unet)
        unet = unets
    return unet

