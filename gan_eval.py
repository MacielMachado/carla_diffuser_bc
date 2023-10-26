from PIL import Image
import torch as th
import numpy as np

from rl_birdview.models.ppo_policy import PpoPolicy
from rl_birdview.models.discriminator import ExpertDataset

import pathlib


if __name__ == '__main__':
    output_dir = pathlib.Path('agent_eval/gan_eval_town2')
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = pathlib.Path('/home/casa/projects/saved_runs/fake-birdview-gan/0/ckpt')
    ckpt_files = ['']
    ckpt_list = [ckpt_file for ckpt_file in ckpt_dir.iterdir() if ckpt_file.stem!='ckpt_latest']
    ckpt_list.sort(key=lambda ckpt_file: int(ckpt_file.stem.split('_')[1]))
    ckpt_idx_list = [0, 8, 16, 24, 32]
    ckpt_list = [ckpt_list[ckpt_idx] for ckpt_idx in ckpt_idx_list]

    expert_loader = th.utils.data.DataLoader(
        ExpertDataset(
            'gail_experts2',
            n_routes=1,
            n_eps=1,
        ),
        batch_size=4,
        shuffle=True,
    )
    birdview_list = [None for _ in range(len(expert_loader.dataset))]
    central_rgb_list = [None for _ in range(len(expert_loader.dataset))]
    left_rgb_list = [None for _ in range(len(expert_loader.dataset))]
    right_rgb_list = [None for _ in range(len(expert_loader.dataset))]
    traj_plot_list = [None for _ in range(len(expert_loader.dataset))]
    for batch in expert_loader:
        obs_dict, _ = batch
        item_idx_array = obs_dict['item_idx'].cpu().numpy()
        for item_idx in range(obs_dict['item_idx'].shape[0]):
            birdview_list[item_idx_array[item_idx]] = obs_dict['birdview'][item_idx]
            central_rgb_list[item_idx_array[item_idx]] = obs_dict['central_rgb'][item_idx]
            left_rgb_list[item_idx_array[item_idx]] = obs_dict['left_rgb'][item_idx]
            right_rgb_list[item_idx_array[item_idx]] = obs_dict['right_rgb'][item_idx]
            traj_plot_list[item_idx_array[item_idx]] = obs_dict['traj_plot'][item_idx]
    birdview_tensor = th.stack(birdview_list)
    central_rgb_tensor = th.stack(central_rgb_list)
    left_rgb_tensor = th.stack(left_rgb_list)
    right_rgb_tensor = th.stack(right_rgb_list)
    traj_plot_tensor = th.stack(traj_plot_list)

    # idx_tensor = th.Tensor([89, 737, 1072, 1287, 1346]).int()
    idx_tensor = th.Tensor([893, 1271, 1775, 1926, 2065]).int()
    birdview_tensor = birdview_tensor.index_select(dim=0, index=idx_tensor)
    central_rgb_tensor = central_rgb_tensor.index_select(dim=0, index=idx_tensor)
    left_rgb_tensor = left_rgb_tensor.index_select(dim=0, index=idx_tensor)
    right_rgb_tensor = right_rgb_tensor.index_select(dim=0, index=idx_tensor)
    traj_plot_tensor = traj_plot_tensor.index_select(dim=0, index=idx_tensor)
    print(ckpt_list)
    for ckpt_idx, ckpt_path in enumerate(ckpt_list):
        saved_variables = th.load(ckpt_path, map_location='cuda')
        policy_kwargs = saved_variables['policy_init_kwargs']
        policy = PpoPolicy(**policy_kwargs)
        policy.load_state_dict(saved_variables['policy_state_dict'])

        fake_birdview_tensor = policy.gan_fake_birdview.fill_expert_dataset(expert_loader)

        fake_birdview_tensor = fake_birdview_tensor.index_select(dim=0, index=idx_tensor)
        img_height = 192
        for img_idx in range(len(idx_tensor)):
            fake_birdview = fake_birdview_tensor[img_idx].numpy()
            fake_birdview = np.transpose(fake_birdview, [1, 2, 0]).astype(np.uint8)
            fake_birdview = Image.fromarray(fake_birdview)

            if img_idx == 0 and ckpt_idx == 0:
                img_width = fake_birdview.size[0]
                img_height = fake_birdview.size[1]
                eval_img = Image.new("RGB", ((len(ckpt_list) + 1) * img_width, len(idx_tensor) * img_height), "white")
                birdview_img = Image.new("RGB", (img_width, len(idx_tensor) * img_height), "white")
                central_rgb_img = Image.new("RGB", (img_width, len(idx_tensor) * img_height), "white")
                left_rgb_img = Image.new("RGB", (img_width, len(idx_tensor) * img_height), "white")
                right_rgb_img = Image.new("RGB", (img_width, len(idx_tensor) * img_height), "white")
                traj_plot_img = Image.new("RGB", (img_width, len(idx_tensor) * img_height), "white")
                epoch_img = Image.new("RGB", (img_width, len(idx_tensor) * img_height), "white")

            if ckpt_idx == 0:
                birdview = birdview_tensor[img_idx].numpy()
                birdview = np.transpose(birdview, [1, 2, 0]).astype(np.uint8)
                birdview = Image.fromarray(birdview)
                birdview_img.paste(birdview, (0, img_idx * img_height))
                eval_img.paste(birdview, (0, img_idx * img_height))

                central_rgb = central_rgb_tensor[img_idx]
                central_rgb = policy.gan_fake_birdview.transforms(central_rgb)
                central_rgb = central_rgb.numpy()
                central_rgb = np.transpose(central_rgb, [1, 2, 0]).astype(np.uint8)
                central_rgb = Image.fromarray(central_rgb)
                central_rgb_img.paste(central_rgb, (0, img_idx * img_height))

                left_rgb = left_rgb_tensor[img_idx]
                left_rgb = policy.gan_fake_birdview.transforms(left_rgb)
                left_rgb = left_rgb.numpy()
                left_rgb = np.transpose(left_rgb, [1, 2, 0]).astype(np.uint8)
                left_rgb = Image.fromarray(left_rgb)
                left_rgb_img.paste(left_rgb, (0, img_idx * img_height))

                right_rgb = right_rgb_tensor[img_idx]
                right_rgb = policy.gan_fake_birdview.transforms(right_rgb)
                right_rgb = right_rgb.numpy()
                right_rgb = np.transpose(right_rgb, [1, 2, 0]).astype(np.uint8)
                right_rgb = Image.fromarray(right_rgb)
                right_rgb_img.paste(right_rgb, (0, img_idx * img_height))

                traj_plot = traj_plot_tensor[img_idx].numpy()
                traj_plot = np.transpose(traj_plot, [1, 2, 0]).astype(np.uint8)
                traj_plot_mask = np.zeros((img_width, img_height, 3), dtype=np.uint8)
                traj_plot_mask[:, :, :] = traj_plot[:, :, :]
                traj_plot = Image.fromarray(traj_plot_mask)
                traj_plot_img.paste(traj_plot, (0, img_idx * img_height))

            eval_img.paste(fake_birdview, ((ckpt_idx + 1) * img_width, img_idx * img_height))
            epoch_img.paste(fake_birdview, (0, img_idx * img_height))

        epoch_img.save(output_dir / '{}.png'.format(ckpt_path.stem))

    eval_img.save(output_dir / 'eval.png')
    birdview_img.save(output_dir / 'birdview.png')
    central_rgb_img.save(output_dir / 'central_rgb.png')
    left_rgb_img.save(output_dir / 'left_rgb.png')
    right_rgb_img.save(output_dir / 'right_rgb.png')
    traj_plot_img.save(output_dir / 'traj_plot.png')
