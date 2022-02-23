import os
import time
import random
import torch
import numpy as np

import mmcv
from mmcv import Config

from mpa.utils.config_utils import MPAConfig
from mpa.utils.logger import config_logger
from mpa.utils import logger

from .registry import STAGES


def _set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_available_types():
    types = []
    for k, v in STAGES.module_dict.items():
        # logger.info(f'key [{k}] = value[{v}]')
        types.append(k)
    return types


# @STAGES.register_module()
class Stage(object):
    def __init__(self, name, mode, config, common_cfg={}, index=0, **kwargs):
        logger.info(f'init stage with: {name}, {mode}, {config}, {common_cfg}, {index}, {kwargs}')
        # the name of 'config' cannot be changed to such as 'config_file'
        # because it is defined as 'config' in recipe file.....
        self.name = name
        self.mode = mode
        self.index = index
        self.input = kwargs.pop('input', {})  # input_map?? input_dict? just input?
        self.output_keys = kwargs.pop('output', [])

        if common_cfg is None:
            common_cfg = dict(output_path='logs')

        if not isinstance(common_cfg, dict):
            raise TypeError(f'common_cfg should be the type of dict but {type(common_cfg)}')
        else:
            if common_cfg.get('output_path') is None:
                logger.info("output_path is not set in common_cfg. set it to 'logs' as default")
                common_cfg['output_path'] = 'logs'

        self.output_prefix = common_cfg['output_path']
        self.output_suffix = f'stage{self.index:02d}_{self.name}'

        # # Work directory
        # work_dir = os.path.join(self.output_prefix, self.output_suffix)
        # mmcv.mkdir_or_exist(os.path.abspath(work_dir))

        if isinstance(config, Config):
            cfg = config
        elif isinstance(config, dict):
            cfg = Config(cfg_dict=config)
        elif isinstance(config, str):
            if os.path.exists(config):
                cfg = MPAConfig.fromfile(config)
            else:
                err_msg = f'cannot find configuration file {config}'
                logger.error(err_msg)
                raise ValueError(err_msg)
        else:
            err_msg = "'config' argument could be one of the \
                       [dictionary, Config object, or string of the cfg file path]"
            logger.error(err_msg)
            raise ValueError(err_msg)

        cfg.merge_from_dict(common_cfg)

        if len(kwargs) > 0:
            addtional_dict = {}
            logger.info('found override configurations for the stage')
            for k, v in kwargs.items():
                addtional_dict[k] = v
                logger.info(f'\t{k}: {v}')
            cfg.merge_from_dict(addtional_dict)

        max_epochs = -1
        if hasattr(cfg, 'total_epochs'):
            max_epochs = cfg.pop('total_epochs')
        if hasattr(cfg, 'runner'):
            if hasattr(cfg.runner, 'max_epochs'):
                if max_epochs != -1:
                    max_epochs = min(max_epochs, cfg.runner.max_epochs)
                else:
                    max_epochs = cfg.runner.max_epochs
        if max_epochs > 0:
            if cfg.runner.max_epochs != max_epochs:
                cfg.runner.max_epochs = max_epochs
                logger.info(f'The maximum number of epochs is adjusted to {max_epochs}.')
            if hasattr(cfg, 'checkpoint_config'):
                if hasattr(cfg.checkpoint_config, 'interval'):
                    if cfg.checkpoint_config.interval > max_epochs:
                        logger.warning(f'adjusted checkpoint interval from {cfg.checkpoint_config.interval} to {max_epochs} \
                            since max_epoch is shorter than ckpt interval configuration')
                        cfg.checkpoint_config.interval = max_epochs

        if hasattr(cfg, 'seed'):
            _set_random_seed(cfg.seed, deterministic=cfg.get('deterministic', False))
        else:
            cfg.seed = None

        # Work directory
        work_dir = cfg.get('work_dir', '')
        work_dir = os.path.join(self.output_prefix, work_dir if work_dir else '', self.output_suffix)
        mmcv.mkdir_or_exist(os.path.abspath(work_dir))
        cfg.work_dir = os.path.abspath(work_dir)
        logger.info(f'work dir = {cfg.work_dir}')

        if not hasattr(cfg, 'gpu_ids'):
            gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            logger.info(f'CUDA_VISIBLE_DEVICES = {gpu_ids}')
            if gpu_ids is not None:
                if isinstance(gpu_ids, str):
                    cfg.gpu_ids = range(len(gpu_ids.split(',')))
                else:
                    raise ValueError(f'not supported type for gpu_ids: {type(gpu_ids)}')
            else:
                cfg.gpu_ids = range(1)

        self.cfg = cfg

    def run(self, **kwargs):
        raise NotImplementedError

    def _init_logger(self, **kwargs):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        config_logger(os.path.join(self.cfg.work_dir, f'{timestamp}.log'), level=self.cfg.log_level)
        logger.info(f'configured logger at {self.cfg.work_dir} with named {timestamp}.log')
        return logger

    @staticmethod
    def configure_data(cfg, training, **kwargs):
        # update data configuration using image options
        def configure_split(target):

            def update_transform(opt, pipeline, idx, transform):
                if isinstance(opt, dict):
                    if '_delete_' in opt.keys() and opt.get('_delete_', False):
                        # if option include _delete_=True, remove this transform from pipeline
                        logger.info(f"configure_data: {transform['type']} is deleted")
                        del pipeline[idx]
                        return
                    logger.info(f"configure_data: {transform['type']} is updated with {opt}")
                    transform.update(**opt)

            def update_config(src, pipeline_options):
                logger.info(f'update_config() {pipeline_options}')
                if src.get('pipeline') is not None or \
                        (src.get('dataset') is not None and src.get('dataset').get('pipeline') is not None):
                    if src.get('pipeline') is not None:
                        pipeline = src.get('pipeline', None)
                    else:
                        pipeline = src.get('dataset').get('pipeline')
                    if isinstance(pipeline, list):
                        for idx, transform in enumerate(pipeline):
                            for opt_key, opt in pipeline_options.items():
                                if transform['type'] == opt_key:
                                    update_transform(opt, pipeline, idx, transform)
                    elif isinstance(pipeline, dict):
                        for _, pipe in pipeline.items():
                            for idx, transform in enumerate(pipe):
                                for opt_key, opt in pipeline_options.items():
                                    if transform['type'] == opt_key:
                                        update_transform(opt, pipe, idx, transform)
                    else:
                        raise NotImplementedError(f'pipeline type of {type(pipeline)} is not supported')
                else:
                    logger.info('no pipeline in the data split')

            split = cfg.data.get(target)
            if split is not None:
                if isinstance(split, list):
                    for sub_item in split:
                        update_config(sub_item, pipeline_options)
                elif isinstance(split, dict):
                    update_config(split, pipeline_options)
                else:
                    logger.warning(f"type of split '{target}'' should be list or dict but {type(split)}")

        logger.info(f'configure_data() {cfg.data}')
        pipeline_options = cfg.data.pop('pipeline_options', None)
        if pipeline_options is not None and isinstance(pipeline_options, dict):
            configure_split('train')
            configure_split('val')
            if not training:
                configure_split('test')
            configure_split('unlabeled')

    @staticmethod
    def get_model_meta(cfg):
        ckpt_path = cfg.get('load_from', None)
        meta = {}
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            meta = ckpt.get('meta', {})
        return meta
