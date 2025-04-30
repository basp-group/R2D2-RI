from data import transforms as T

import torch


def forward(
    layers,
    i,
    net,
    res_n,
    output_n,
    mean,
    eps=1e-110,
    dirty=None,
    PSF=None,
    input_order="res_rec",
):

    if layers == 1:
        if i == 0:
            output = net(torch.cat((res_n, output_n), dim=1)) + output_n
        else:
            match input_order:
                case "res_rec":
                    output = net(torch.cat((res_n, output_n), dim=1)) + output_n
                case "rec_res":
                    output = net(torch.cat((output_n, res_n), dim=1)) + output_n
    else:
        for j in range(layers):
            if j == 0:
                output = net[j](torch.cat((res_n, output_n), dim=1)) + output_n
            else:
                output = net[j](torch.cat((res_tmp, output), dim=1)) + output
                del res_tmp
            if j < (layers - 1):
                output *= mean + 1e-110

                res_tmp = get_residual_psf(dirty, output, PSF).to(torch.float32)
                output, mean = T.normalize_instance(output, eps=1e-110)
                res_tmp = T.normalize(res_tmp, mean, eps=1e-110)
    output = torch.clip(output * (mean + eps), min=0, max=None)
    return output


def get_residual_psf(dirty, output, psf):
    if psf.size(-1) // output.size(-1) == 2:
        start, end = output.size(-2), output.size(-2) * 2
        psf_fft = torch.fft.fft2(psf, dim=(-2, -1))
    elif psf.size(-1) // output.size(-1) == 1:
        start, end = int(output.size(-2) / 2), int(output.size(-2) * 1.5)
        psf_fft = torch.fft.fft2(psf, s=(psf.size(-2), psf.size(-1)), dim=(-2, -1))

    output_fft = torch.fft.fft2(output, s=(psf.size(-2), psf.size(-1)), dim=(-2, -1))
    output_dirty = torch.fft.ifft2(output_fft * psf_fft, dim=(-2, -1))
    residual = dirty - torch.real(output_dirty)[..., start:end, start:end]

    return residual
