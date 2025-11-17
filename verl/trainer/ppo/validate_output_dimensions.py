"""
验证 generate_sequences 输出维度的工具函数
用于比较恢复后的 gen_batch_output 和原始备份 gen_batch_output_ori 的维度
"""

import numpy as np
from verl import DataProto


def validate_output_dimensions(gen_batch_output: DataProto, gen_batch_output_ori: DataProto, verbose: bool = True) -> dict:
    """
    验证 gen_batch_output 的维度是否正确，与备份的 gen_batch_output_ori 进行比较。
    
    Args:
        gen_batch_output: 恢复后的输出
        gen_batch_output_ori: 原始备份的输出
        verbose: 是否打印详细信息
        
    Returns:
        dict: 包含验证结果的字典
        {
            'is_valid': bool,  # 是否所有维度都匹配
            'batch_size_match': bool,  # batch size 是否匹配
            'batch_size': (int, int),  # (output, original)
            'tensor_shapes_match': dict,  # 每个tensor的shape是否匹配
            'non_tensor_shapes_match': dict,  # 每个non_tensor的shape是否匹配
            'errors': list,  # 错误信息列表
        }
    """
    result = {
        'is_valid': True,
        'batch_size_match': False,
        'batch_size': (None, None),
        'tensor_shapes_match': {},
        'non_tensor_shapes_match': {},
        'errors': [],
    }
    
    # 1. 检查 batch size (使用 __len__)
    len_output = len(gen_batch_output)
    len_ori = len(gen_batch_output_ori)
    result['batch_size'] = (len_output, len_ori)
    result['batch_size_match'] = (len_output == len_ori)
    
    if not result['batch_size_match']:
        result['is_valid'] = False
        result['errors'].append(
            f"Batch size mismatch: output={len_output}, original={len_ori}"
        )
    
    # 2. 检查 batch.batch_size
    batch_size_output = gen_batch_output.batch.batch_size
    batch_size_ori = gen_batch_output_ori.batch.batch_size
    
    if batch_size_output != batch_size_ori:
        result['is_valid'] = False
        result['errors'].append(
            f"batch.batch_size mismatch: output={batch_size_output}, original={batch_size_ori}"
        )
    
    # 3. 检查每个 tensor 的 shape
    output_keys = set(gen_batch_output.batch.keys())
    ori_keys = set(gen_batch_output_ori.batch.keys())
    
    # 检查是否有缺失或多余的 keys
    missing_keys = ori_keys - output_keys
    extra_keys = output_keys - ori_keys
    
    if missing_keys:
        result['is_valid'] = False
        result['errors'].append(f"Missing tensor keys in output: {missing_keys}")
    
    if extra_keys:
        result['is_valid'] = False
        result['errors'].append(f"Extra tensor keys in output: {extra_keys}")
    
    # 检查共同 keys 的 shape
    common_keys = output_keys & ori_keys
    for key in common_keys:
        tensor_output = gen_batch_output.batch[key]
        tensor_ori = gen_batch_output_ori.batch[key]
        
        shape_output = tensor_output.shape
        shape_ori = tensor_ori.shape
        
        # 检查 shape 是否匹配
        if shape_output != shape_ori:
            result['is_valid'] = False
            result['tensor_shapes_match'][key] = False
            result['errors'].append(
                f"Tensor '{key}' shape mismatch: output={shape_output}, original={shape_ori}"
            )
        else:
            result['tensor_shapes_match'][key] = True
            
        # 检查 dtype 是否匹配
        if tensor_output.dtype != tensor_ori.dtype:
            result['is_valid'] = False
            result['errors'].append(
                f"Tensor '{key}' dtype mismatch: output={tensor_output.dtype}, original={tensor_ori.dtype}"
            )
    
    # 4. 检查 non_tensor_batch 的 shape
    output_non_keys = set(gen_batch_output.non_tensor_batch.keys())
    ori_non_keys = set(gen_batch_output_ori.non_tensor_batch.keys())
    
    missing_non_keys = ori_non_keys - output_non_keys
    extra_non_keys = output_non_keys - ori_non_keys
    
    if missing_non_keys:
        result['is_valid'] = False
        result['errors'].append(f"Missing non_tensor keys in output: {missing_non_keys}")
    
    if extra_non_keys:
        result['is_valid'] = False
        result['errors'].append(f"Extra non_tensor keys in output: {extra_non_keys}")
    
    # 检查共同 non_tensor keys 的 shape
    common_non_keys = output_non_keys & ori_non_keys
    for key in common_non_keys:
        arr_output = gen_batch_output.non_tensor_batch[key]
        arr_ori = gen_batch_output_ori.non_tensor_batch[key]
        
        # 对于 numpy array
        if isinstance(arr_output, np.ndarray) and isinstance(arr_ori, np.ndarray):
            shape_output = arr_output.shape
            shape_ori = arr_ori.shape
            
            if shape_output != shape_ori:
                result['is_valid'] = False
                result['non_tensor_shapes_match'][key] = False
                result['errors'].append(
                    f"Non-tensor '{key}' shape mismatch: output={shape_output}, original={shape_ori}"
                )
            else:
                result['non_tensor_shapes_match'][key] = True
                
            # 检查 dtype
            if arr_output.dtype != arr_ori.dtype:
                result['is_valid'] = False
                result['errors'].append(
                    f"Non-tensor '{key}' dtype mismatch: output={arr_output.dtype}, original={arr_ori.dtype}"
                )
        else:
            # 对于其他类型（如 list），检查长度
            len_output = len(arr_output) if hasattr(arr_output, '__len__') else None
            len_ori = len(arr_ori) if hasattr(arr_ori, '__len__') else None
            
            if len_output != len_ori:
                result['is_valid'] = False
                result['non_tensor_shapes_match'][key] = False
                result['errors'].append(
                    f"Non-tensor '{key}' length mismatch: output={len_output}, original={len_ori}"
                )
            else:
                result['non_tensor_shapes_match'][key] = True
    
    # 5. 打印结果
    if verbose:
        print("\n" + "="*80)
        print("DataProto 维度验证结果")
        print("="*80)
        print(f"Batch size: output={len_output}, original={len_ori}, match={result['batch_size_match']}")
        print(f"\nTensor shapes match: {sum(result['tensor_shapes_match'].values())}/{len(result['tensor_shapes_match'])}")
        print(f"Non-tensor shapes match: {sum(result['non_tensor_shapes_match'].values())}/{len(result['non_tensor_shapes_match'])}")
        
        if result['errors']:
            print(f"\n发现 {len(result['errors'])} 个错误:")
            for i, error in enumerate(result['errors'], 1):
                print(f"  {i}. {error}")
        else:
            print("\n✓ 所有维度匹配!")
        print("="*80 + "\n")
    
    return result


def print_data_proto_summary(data: DataProto, name: str = "DataProto"):
    """
    打印 DataProto 的维度摘要信息。
    
    Args:
        data: DataProto 对象
        name: 名称标识
    """
    print(f"\n{name} 维度摘要:")
    print(f"  Batch size: {len(data)}")
    print(f"  batch.batch_size: {data.batch.batch_size}")
    
    print(f"\n  Tensor keys ({len(data.batch.keys())}):")
    for key in sorted(data.batch.keys()):
        tensor = data.batch[key]
        print(f"    {key}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    print(f"\n  Non-tensor keys ({len(data.non_tensor_batch.keys())}):")
    for key in sorted(data.non_tensor_batch.keys()):
        val = data.non_tensor_batch[key]
        if isinstance(val, np.ndarray):
            print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"    {key}: type={type(val).__name__}, len={len(val) if hasattr(val, '__len__') else 'N/A'}")

