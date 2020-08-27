from typing import Iterable
import random
import torchvision.transforms as transforms
from argparse import Namespace
import numpy as np
from PIL import Image


def make_power_2(n, base=32.0):
    return int(round(n / base) * base)


def get_img_params(opt: Namespace, size: Iterable):
    w, h = size
    new_h, new_w = h, w
    # resize image to be loadSize x loadSize
    if opt.dataset_scale == 'resize':
        new_h = new_w = opt.load_size
    # scale image width to be loadSize
    elif opt.dataset_scale == 'scale-width':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    # scale image height to be loadSize:
    elif opt.dataset_scale == 'scale-height':
        new_h = opt.load_size
        new_w = opt.load_size * w // h
    # randomly scale image width to be somewhere between loadSize and fineSize
    elif opt.dataset_scale == 'random-scale-width':
        new_w = random.randint(opt.fine_size, opt.load_size + 1)
        new_h = new_w * h // w
    # randomly scale image height to be somewhere between loadSize and fineSize
    elif opt.dataset_scale == 'random-scale-height':
        new_h = random.randint(opt.fine_size, opt.load_size + 1)
        new_w = new_h * w // h
    new_w = int(round(new_w / 4)) * 4
    new_h = int(round(new_h / 4)) * 4

    crop_x = crop_y = 0
    crop_w = crop_h = 0
    if opt.dataset_crop == 'crop' or opt.dataset_crop == 'scaled-crop':
        if 'crop' in opt.dataset_crop:  # crop patches of size fineSize x fineSize
            crop_w = crop_h = opt.fine_size
        else:
            if 'width' in opt.dataset_scale:  # crop patches of width fineSize
                crop_w = opt.fine_size
                crop_h = opt.fine_size * h // w
            else:  # crop patches of height fineSize
                crop_h = opt.fine_size
                crop_w = opt.fine_size * w // h

        crop_w, crop_h = make_power_2(crop_w), make_power_2(crop_h)
        x_span = (new_w - crop_w) // 2
        crop_x = np.maximum(0, np.minimum(x_span * 2, int(np.random.randn() * x_span / 3 + x_span)))
        crop_y = random.randint(0, np.minimum(np.maximum(0, new_h - crop_h), new_h // 8))
        # crop_x = random.randint(0, np.maximum(0, new_w - crop_w))
        # crop_y = random.randint(0, np.maximum(0, new_h - crop_h))
    else:
        new_w, new_h = make_power_2(new_w), make_power_2(new_h)

    flip = (random.random() > 0.5) and (opt.dataset_mode != 'pose')
    return {'new_size': (new_w, new_h), 'crop_size': (crop_w, crop_h), 'crop_pos': (crop_x, crop_y), 'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    # resize input image
    if opt.dataset_scale == 'resize':
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Scale(osize, method))
    else:
        transform_list.append(transforms.Lambda(lambda img: __scale_image(img, params['new_size'], method)))

    # crop patches from image
    if 'crop' in opt.dataset_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_size'], params['crop_pos'])))

    # random flip
    if opt.is_train and opt.flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def toTensor_normalize():
    transform_list = [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_image(img, size, method=Image.BICUBIC):
    w, h = size
    return img.resize((w, h), method)


def __crop(img, size, pos):
    ow, oh = img.size
    tw, th = size
    x1, y1 = pos
    if ow > tw or oh > th:
        return img.crop((x1, y1, min(ow, x1 + tw), min(oh, y1 + th)))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def get_video_params(opt, n_frames_total, cur_seq_len, index):
    tG = opt.n_input_gen_frames
    if opt.is_train:
        n_frames_total = min(n_frames_total, cur_seq_len - tG + 1)

        n_gpus = opt.gen_gpus if opt.batch_size == 1 else 1  # number of generator GPUs for each batch
        n_frames_per_load = opt.max_frames_per_gpu * n_gpus  # number of frames to load into GPUs at one time (for each batch)
        n_frames_per_load = min(n_frames_total, n_frames_per_load)
        n_loadings = n_frames_total // n_frames_per_load  # how many times are needed to load entire sequence into GPUs
        n_frames_total = n_frames_per_load * n_loadings + tG - 1  # rounded overall number of frames to read from the sequence

        max_t_step = min(opt.max_t_step, (cur_seq_len - 1) // (n_frames_total - 1))
        t_step = np.random.randint(max_t_step) + 1  # spacing between neighboring sampled frames
        offset_max = max(1, cur_seq_len - (
                    n_frames_total - 1) * t_step)  # maximum possible index for the first frame
        if opt.dataset_mode == 'pose':
            start_idx = index % offset_max
        else:
            start_idx = np.random.randint(offset_max)  # offset for the first frame to load
        if opt.debug:
            print("loading %d frames in total, first frame starting at index %d, space between neighboring frames is %d"
                  % (n_frames_total, start_idx, t_step))
    else:
        n_frames_total = tG
        start_idx = index
        t_step = 1
    return n_frames_total, start_idx, t_step


def concat_frame(A, Ai, nF):
    if A is None:
        A = Ai
    else:
        c = Ai.size()[0]
        if A.size()[0] == nF * c:
            A = A[c:]
        A = torch.cat([A, Ai])
    return A