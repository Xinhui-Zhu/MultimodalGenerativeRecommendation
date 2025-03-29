# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

from cProfile import run
from logging import getLogger
import torch
import json
from REC.data import *
from REC.config import Config
from REC.utils import init_logger, get_model, init_seed, set_color
from REC.trainer import Trainer
import torch.distributed as dist

import os
import numpy as np
import argparse
import torch.distributed as dist
import torch


def convert_str(s):
    try:
        if s.lower() == 'none':
            return None
        if s.lower() == 'true':
            return True
        if s.lower() == 'false':
            return False
        float_val = float(s)
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except ValueError:
        print(f"Unable to convert the string '{s}' to None / Bool / Float / Int, retaining the original string.")
        return s


def run_loop(local_rank, config_file=None, saved=True, extra_args=[]):

    # configurations initialization
    config = Config(config_file_list=config_file)

    device = torch.device("cuda", local_rank)
    config['device'] = device
    if len(extra_args):
        for i in range(0, len(extra_args), 2):
            key = extra_args[i][2:]
            value = extra_args[i + 1]
            try:
                if '[' in value or '{' in value:
                    value = json.loads(value)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            value[k] = convert_str(v)
                    else:
                        value = [convert_str(x) for x in value]
                else:
                    value = convert_str(value)
                if '.' in key:
                    k1, k2 = key.split('.')
                    config[k1][k2] = value
                else:
                    config[key] = value
            except:
                raise ValueError(f"{key} {value} invalid")

    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    if 'text_path' in config:
        config['text_path'] = os.path.join(config['text_path'], config['dataset'] + '.csv')
        logger.info(f"Update text_path to {config['text_path']}")

    # get model and data
    if config['split_data']:
        dataload = load_data(config)
        original_dataset = config['dataset']
        print(config['dataset'])
        config['dataset'] = original_dataset + "-train"
        train_dataload = load_data(config)
        config['dataset'] = original_dataset + "-test"
        test_dataload = load_data(config)
        config['dataset'] = original_dataset 
        train_loader, _, _ = bulid_dataloader(config, train_dataload)
        _, valid_loader, test_loader = bulid_dataloader(config, test_dataload)
        print(f"len(train_loader) = {len(train_loader)}")
        print(f"len(test_loader) = {len(test_loader)}")
        logger.info(train_dataload)
        logger.info(test_dataload)
    else:
        dataload = load_data(config)
        train_loader, valid_loader, test_loader = bulid_dataloader(config, dataload)
        print(f"{len(train_loader)} =")
        logger.info(dataload)

    model = get_model(config['model'])(config, dataload)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    world_size = torch.distributed.get_world_size()
    trainer = Trainer(config, model)

    logger.info(set_color('\nWorld_Size', 'pink') + f' = {world_size} \n')
    logger.info(config)
    logger.info(model)

    if config['val_only']:
        # ckpt_path = os.path.join(config['checkpoint_dir'], 'pytorch_model.bin')
        # ckpt = torch.load(ckpt_path, map_location='cpu')
        # logger.info(f'Eval only model load from {ckpt_path}')
        # msg = trainer.model.load_state_dict(ckpt, False)
        # logger.info(f'{msg.unexpected_keys = }')
        # logger.info(f'{msg.missing_keys = }')
        model_file = os.path.join(config['checkpoint_dir'], 'HLLM-0.pth/checkpoint/zero_pp_rank_0_mp_rank_00_model_states.pt')
        result_dict = trainer.evaluate(test_loader, load_best_model=True, show_progress=config['show_progress'], init_model=True)
        test_result = result_dict.get("all", None)
        test_result_cold = result_dict.get("cold", None)
        test_result_warm = result_dict.get("warm", None)
        test_result_strictly_cold = result_dict.get("strictly_cold", None)

        logger.info(set_color('test result', 'yellow') + f': {test_result}')
        if test_result_cold:
            logger.info(set_color('test result strictly cold', 'yellow') + f': {test_result_strictly_cold}')
            logger.info(set_color('test result cold', 'yellow') + f': {test_result_cold}')
            logger.info(set_color('test result warm', 'yellow') + f': {test_result_warm}')
    else:
        # training process
        best_valid_score, best_valid_result = trainer.fit(
            train_loader, valid_loader, saved=saved, show_progress=config['show_progress']
        )
        logger.info(f'Trianing Ended' + set_color('best valid ', 'yellow') + f': {best_valid_result}')

        # model evaluation
        result_dict = trainer.evaluate(test_loader, load_best_model=saved, show_progress=config['show_progress'])
        test_result = result_dict.get("all", None)
        test_result_cold = result_dict.get("cold", None)
        test_result_warm = result_dict.get("warm", None)
        test_result_strictly_cold = result_dict.get("strictly_cold", None)

        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')
        if test_result_cold:
            logger.info(set_color('test result strictly cold', 'yellow') + f': {test_result_strictly_cold}')
            logger.info(set_color('test result cold', 'yellow') + f': {test_result_cold}')
            logger.info(set_color('test result warm', 'yellow') + f': {test_result_warm}')

        return {
            'best_valid_score': best_valid_score,
            'valid_score_bigger': config['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs='+', type=str)
    args, extra_args = parser.parse_known_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    config_file = args.config_file

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    try:
        run_loop(local_rank=local_rank, config_file=config_file, extra_args=extra_args)
    except torch.cuda.OutOfMemoryError as e:
        print("CUDA OutOfMemoryError encountered!")
        print(torch.cuda.memory_summary(device=0))  # 打印显存摘要信息
        torch.cuda.empty_cache()  # 清理显存
        raise e  # 重新抛出错误以便退出程序或进一步处理
