import json
import time

from json.decoder import JSONDecodeError


def check_raise(file_path='raise_flag.json'):
    file_raise_flag = False
    while True:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_raise_flag = json.load(f)['raise_flag']
        except JSONDecodeError:
            time.sleep(1)
        else:
            break
    if file_raise_flag:
        raise Exception('[tmp log] actor group raise error')
