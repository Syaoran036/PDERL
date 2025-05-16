import pprint
import torch
import os
import json

class Parameters:
    def __init__(self, cfg, init=True):
        if not init:
            return
        # 设置运行设备
        if not cfg.get('disable_cuda', False) and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # 从传入的字典获取属性
        for attr, value in cfg.items():
            if attr == 'actor' or attr == 'critic':
                for sub_attr, sub_value in value.items():
                    setattr(self, f"{attr}_{sub_attr}", sub_value)
            else:
                setattr(self, attr, value)

        # 根据环境设置帧数
        if hasattr(self, 'env'):
            if self.env == 'Hopper-v2':
                self.num_frames = 4000000
            elif self.env in ['Ant-v2', 'Walker2d-v2', 'HalfCheetah-v2']:
                self.num_frames = 6000000
            else:
                self.num_frames = 2000000

            # 同步设置
            if self.env in ['Hopper-v2', 'Ant-v2', 'Walker2d-v2']:
                self.rl_to_ea_synch_period = 1
            else:
                self.rl_to_ea_synch_period = 10

            # 如果命令行传递了同步周期，则覆盖默认值
            if hasattr(self, 'sync_period') and self.sync_period is not None:
                self.rl_to_ea_synch_period = self.sync_period

            # 根据环境设置试验次数
            if self.env in ['Hopper-v2', 'Reacher-v2']:
                self.num_evals = 3
            elif self.env == 'Walker2d-v2':
                self.num_evals = 5
            else:
                self.num_evals = 1

            # 根据环境设置精英率
            if self.env in ['Reacher-v2', 'Walker2d-v2', 'Ant-v2', 'Hopper-v2']:
                self.elite_fraction = 0.2
            else:
                self.elite_fraction = 0.1

        # 计算学习起始步数
        if hasattr(self, 'buffer_size') and hasattr(self, 'batch_size'):
            self.learn_start = (1 + self.buffer_size / self.batch_size) * 2
        self.total_steps = getattr(self, 'num_frames', 2000000)

        # 保存结果相关设置
        self.state_dim = None  # 外部初始化
        self.action_dim = None  # 外部初始化
        if hasattr(self, 'logdir') and not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

    def write_params(self, stdout=True):
        # 将所有超参数写入文件
        params = pprint.pformat(vars(self), indent=4)
        if stdout:
            print(params)

        with open(os.path.join(self.logdir, 'info.txt'), 'a') as f:
            f.write(params)