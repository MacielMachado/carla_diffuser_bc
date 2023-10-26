
from rl_birdview.models.gan_layers import GanFakeBirdview
from rl_birdview.models.discriminator import ExpertDataset
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import torch as th
import torchvision.transforms as transforms


if __name__ == '__main__':
    expert_loader = torch.utils.data.DataLoader(
        ExpertDataset(
            'gail_experts',
            n_routes=1,
            n_eps=1,
        ),
        batch_size=16,
        shuffle=True,
    )

    transforms_image = transforms.Compose([
        transforms.Resize((192, 192)),
    ])
    
    # gan_fake_birdview = GanFakeBirdview()
    # generator_variables = torch.load('saved_models/fake_birdview/generator_8.pth', map_location='cuda')
    # gan_fake_birdview.generator.load_state_dict(generator_variables)
    # fake_birdview_tensor = gan_fake_birdview.fill_expert_dataset(expert_loader)
    route_id = 0
    ep_id = 0
    i_step = 0
    expert_file_dir = Path('gail_experts')
    traj_length = 0
    central_rgb_list = [None for _ in range(len(expert_loader.dataset))]
    left_rgb_list = [None for _ in range(len(expert_loader.dataset))]
    right_rgb_list = [None for _ in range(len(expert_loader.dataset))]
    traj_plot_list = [None for _ in range(len(expert_loader.dataset))]
    for batch in expert_loader:
        obs_dict, _ = batch
        item_idx_array = obs_dict['item_idx'].cpu().numpy()
        for item_idx in range(obs_dict['item_idx'].shape[0]):
            central_rgb_list[item_idx_array[item_idx]] = obs_dict['central_rgb'][item_idx]
            left_rgb_list[item_idx_array[item_idx]] = obs_dict['left_rgb'][item_idx]
            right_rgb_list[item_idx_array[item_idx]] = obs_dict['right_rgb'][item_idx]
            traj_plot_list[item_idx_array[item_idx]] = obs_dict['traj_plot'][item_idx]

    central_rgb_tensor = th.stack(central_rgb_list)
    left_rgb_tensor = th.stack(left_rgb_list)
    right_rgb_tensor = th.stack(right_rgb_list)
    traj_plot_tensor = th.stack(traj_plot_list)

    for img_idx in range(traj_plot_tensor.shape[0]):
        if i_step >= traj_length:
            episode_dir = expert_file_dir / ('route_%02d' % route_id) / ('ep_%02d' % ep_id)
            # (episode_dir / 'central_rgb2').mkdir(parents=True)
            # (episode_dir / 'left_rgb2').mkdir(parents=True)
            # (episode_dir / 'right_rgb2').mkdir(parents=True)
            (episode_dir / 'traj_plot').mkdir(parents=True)
            route_df = pd.read_json(episode_dir / 'episode.json')
            traj_length = route_df.shape[0]
            route_id += 1
            i_step = 0

        # central_rgb = central_rgb_tensor[img_idx]
        # central_rgb = transforms_image(central_rgb)
        # central_rgb = central_rgb.numpy()
        # central_rgb = np.transpose(central_rgb, [1, 2, 0]).astype(np.uint8)
        # Image.fromarray(central_rgb).save(episode_dir / 'central_rgb2' / '{:0>4d}.png'.format(i_step))

        # left_rgb = left_rgb_tensor[img_idx]
        # left_rgb = transforms_image(left_rgb)
        # left_rgb = left_rgb.numpy()
        # left_rgb = np.transpose(left_rgb, [1, 2, 0]).astype(np.uint8)
        # Image.fromarray(left_rgb).save(episode_dir / 'left_rgb2' / '{:0>4d}.png'.format(i_step))

        # right_rgb = right_rgb_tensor[img_idx]
        # right_rgb = transforms_image(right_rgb)
        # right_rgb = right_rgb.numpy()
        # right_rgb = np.transpose(right_rgb, [1, 2, 0]).astype(np.uint8)
        # Image.fromarray(right_rgb).save(episode_dir / 'right_rgb2' / '{:0>4d}.png'.format(i_step))

        traj_plot = traj_plot_tensor[img_idx].numpy()
        traj_plot = np.transpose(traj_plot, [1, 2, 0]).astype(np.uint8)
        Image.fromarray(traj_plot).save(episode_dir / 'traj_plot' / '{:0>4d}.png'.format(i_step))
        i_step += 1
