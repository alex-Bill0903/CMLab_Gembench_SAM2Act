import os
import argparse

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from flask import Flask, request

from challenges.actioner import RandomActioner, ThreeDLotusPlusActioner
from genrobo3d.evaluation.eval_simple_policy_Challenge import Actioner


from genrobo3d.evaluation.eval_simple_policy_server import ServerArguments
import json
# def load_args_from_file(path: str):
#     with open(path, 'r') as f:
#         cfg = json.load(f)
#     # 用 Tap 先 parse 一次空 args 以建立預設屬性
#     args = ServerArguments().parse_args(known_only=True)
#     # 把 config 裡的所有鍵值都覆蓋到 args 上
#     for k, v in cfg.items():
#         setattr(args, k, v)
#     return args

def load_args_from_file(path: str):
    # 1) 讀 JSON
    with open(path, 'r') as f:
        cfg = json.load(f)

    # 2) 把 JSON 裡的每個欄位，build 成 Parser 能接受的 args list
    args_list = []
    for k, v in cfg.items():
        flag = f'--{k}'
        # 布林值 ：True -> 加上 flag, False -> 不加
        if isinstance(v, bool):
            if v:
                args_list.append(flag)
        # List: --key v1 v2 v3
        elif isinstance(v, list):
            args_list.append(flag)
            args_list += [str(x) for x in v]
        # 其他型別：--key value
        else:
            args_list += [flag, str(v)]

    # 3) 交給 Tap 解析；known_only=True 讓它忽略 JSON 裡沒有在 class 定義的欄位
    args = ServerArguments().parse_args(args_list, known_only=True)
    # 4) 把 extra 未定義參數收集起來（原本程式邏輯）
    args.remained_args = getattr(args, 'extra_args', [])

    return args


def main(args):
    app = Flask(__name__)
    
    # actioner = RandomActioner()
    # actioner = ThreeDLotusPlusActioner()
    my_args_config = load_args_from_file('challenges/args_config.json')
    my_args_config.exp_config = os.path.join(my_args_config.expr_dir, 'logs', 'training_config.yaml')
    my_args_config.checkpoint = os.path.join(
        my_args_config.expr_dir, 'ckpts', f'model_step_{my_args_config.ckpt_step}.pt'
    )
    actioner = Actioner(my_args_config)

    @app.route('/predict', methods=['POST'])
    def predict():
        '''
        batch is a dict containing:
            taskvar: str, 'task+variation'
            episode_id: int
            step_id: int, [0, 25]
            instruction: str
            obs_state_dict: observations from genrobo3d.rlbench.environments.RLBenchEnv 
        '''
        data = request.data
        batch = msgpack_numpy.unpackb(data)

        action = actioner.predict(**batch)
        # print('action', action)

        action = msgpack_numpy.packb(action)
        return action
    
    app.run(host=args.ip, port=args.port, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Actioner server')
    parser.add_argument('--ip', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=13000)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    main(args)

