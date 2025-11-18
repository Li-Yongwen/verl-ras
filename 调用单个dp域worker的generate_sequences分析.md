# 调用单个DP域Worker执行generate_sequences的代码修改分析

## 一、需求理解

**目标**：让 WorkerGroup 只调用单个 DP（数据并行）域的 worker 执行 `generate_sequences` 方法，而不是所有 DP 域的 worker。

**当前机制**：
- `generate_sequences` 使用 `@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))` 装饰器
- 该 dispatch 模式会将数据分发到所有 DP 域的 workers
- 所有 workers 都会执行 `generate_sequences`

## 二、核心架构分析

### 2.1 当前数据流

```
调用 generate_sequences
    ↓
dispatch_lazy_compute_data_proto (mesh_name="rollout")
    ↓
查询 worker_group._dispatch_info["rollout"] 获取 dp_rank_mapping
    ↓
dispatch_nd_compute_dataproto (根据 dp_rank_mapping 分发数据到所有 DP 域)
    ↓
execute_all (执行所有 workers)
    ↓
collect_lazy_compute_data_proto (收集所有 DP 域的结果)
```

### 2.2 关键代码位置

#### 1. generate_sequences 方法定义
**文件**: `verl/workers/fsdp_workers.py`
**位置**: 第 1233-1308 行

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
@DistProfiler.annotate(color="red", role="rollout_generate")
def generate_sequences(self, prompts: DataProto):
    # ... 实现代码
```

#### 2. Dispatch 机制
**文件**: `verl/single_controller/base/decorator.py`
**位置**: 第 277-315 行

- `dispatch_lazy_compute_data_proto`: 根据 mesh_name 查询 dp_rank_mapping，然后分发数据
- `collect_lazy_compute_data_proto`: 根据 collect_mask 收集结果

#### 3. Execute 机制
**文件**: `verl/single_controller/ray/base.py`
**位置**: 第 615-655 行

- `execute_all`: 执行所有 workers 的方法

## 三、需要修改的代码部分

### 方案一：创建新的 Dispatch 模式（推荐）

创建一个专门用于单个 DP 域的 dispatch 模式，这样可以保持代码的清晰性和可维护性。

#### 3.1 修改 `verl/single_controller/base/decorator.py`

**新增函数**：

```python
def dispatch_single_dp_domain_dataproto(mesh_name, target_dp_rank, worker_group, *args, **kwargs):
    """
    只分发数据到指定 DP 域的 workers。
    
    Args:
        mesh_name: mesh 名称（如 "rollout"）
        target_dp_rank: 目标 DP rank（只使用该 DP 域的 workers）
        worker_group: WorkerGroup 实例
        *args, **kwargs: 要分发的数据
    """
    from verl.single_controller.base.worker_group import WorkerGroup
    
    assert isinstance(worker_group, WorkerGroup)
    
    # 查询 dispatch info
    if mesh_name not in worker_group._dispatch_info:
        worker_group._dispatch_info[mesh_name] = worker_group._query_dispatch_info(mesh_name)
        assert len(worker_group._dispatch_info[mesh_name]) == worker_group.world_size
    
    dp_rank_mapping = worker_group._dispatch_info[mesh_name]
    
    # 筛选出目标 DP rank 的 workers
    target_worker_indices = [
        i for i, dp_rank in enumerate(dp_rank_mapping) 
        if dp_rank == target_dp_rank
    ]
    
    if not target_worker_indices:
        raise ValueError(f"没有找到 dp_rank={target_dp_rank} 的 workers")
    
    # 只分发数据到目标 workers
    # 注意：这里需要修改 dispatch_nd_compute_dataproto 的逻辑
    # 或者创建一个新的分发函数
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(
        len(target_worker_indices), *args, **kwargs
    )
    
    # 创建新的映射：只包含目标 workers
    filtered_dp_rank_mapping = [dp_rank_mapping[i] for i in target_worker_indices]
    
    return dispatch_nd_compute(filtered_dp_rank_mapping, 1, worker_group, *splitted_args, **splitted_kwargs)


def collect_single_dp_domain_dataproto(mesh_name, target_dp_rank, worker_group, output):
    """
    只收集指定 DP 域的 workers 的结果。
    """
    from verl.single_controller.base.worker_group import WorkerGroup
    
    assert isinstance(worker_group, WorkerGroup)
    
    # 查询 dispatch info 和 collect info
    if mesh_name not in worker_group._dispatch_info:
        worker_group._dispatch_info[mesh_name] = worker_group._query_dispatch_info(mesh_name)
    
    if mesh_name not in worker_group._collect_info:
        worker_group._collect_info[mesh_name] = worker_group._query_collect_info(mesh_name)
    
    dp_rank_mapping = worker_group._dispatch_info[mesh_name]
    collect_mask = worker_group._collect_info[mesh_name]
    
    # 筛选出目标 DP rank 的 workers
    target_worker_indices = [
        i for i, dp_rank in enumerate(dp_rank_mapping) 
        if dp_rank == target_dp_rank
    ]
    
    # 只收集目标 workers 的结果
    filtered_output = [output[i] for i in target_worker_indices]
    filtered_collect_mask = [collect_mask[i] for i in target_worker_indices]
    
    return collect_nd_compute_dataproto(filtered_collect_mask, worker_group, filtered_output)


def make_single_dp_domain_dispatch_fn(mesh_name, target_dp_rank):
    """
    创建单个 DP 域的 dispatch 函数。
    """
    return {
        "dispatch_fn": partial(dispatch_single_dp_domain_dataproto, mesh_name, target_dp_rank),
        "collect_fn": partial(collect_single_dp_domain_dataproto, mesh_name, target_dp_rank),
    }
```

**问题**：这个方案需要修改 `execute_all` 的逻辑，使其只执行目标 workers。需要进一步修改。

#### 3.2 修改 `verl/single_controller/ray/base.py`

需要添加一个新的 execute 方法，只执行特定 workers：

```python
def execute_selected_workers(self, method_name: str, worker_indices: list[int], *args, **kwargs):
    """
    只执行指定索引的 workers 的方法。
    
    Args:
        method_name: 要执行的方法名
        worker_indices: 要执行的 worker 索引列表
        *args, **kwargs: 方法参数
    """
    selected_workers = [self._workers[i] for i in worker_indices]
    selected_args = [[args[i] for i in worker_indices] for args in args]
    selected_kwargs = {k: [v[i] for i in worker_indices] for k, v in kwargs.items()}
    
    futures = []
    for worker, worker_args, worker_kwargs in zip(selected_workers, selected_args, selected_kwargs):
        method = getattr(worker, method_name)
        future = method.remote(*worker_args, **worker_kwargs)
        futures.append(future)
    
    return futures
```

### 方案二：修改现有 Dispatch 函数（更简单）

直接修改 `dispatch_lazy_compute_data_proto` 和相关的 execute 函数，添加一个参数来指定目标 DP rank。

#### 3.1 修改 `verl/single_controller/base/decorator.py`

修改 `make_nd_compute_dataproto_dispatch_fn` 函数，添加可选的 `target_dp_rank` 参数：

```python
def make_nd_compute_dataproto_dispatch_fn(mesh_name, target_dp_rank=None):
    """
    创建 ND compute dataproto dispatch 函数。
    
    Args:
        mesh_name: mesh 名称
        target_dp_rank: 如果指定，只使用该 DP rank 的 workers；如果为 None，使用所有 workers
    """
    if target_dp_rank is not None:
        return {
            "dispatch_fn": partial(dispatch_single_dp_lazy_compute_data_proto, mesh_name, target_dp_rank),
            "collect_fn": partial(collect_single_dp_lazy_compute_data_proto, mesh_name, target_dp_rank),
        }
    else:
        return {
            "dispatch_fn": partial(dispatch_lazy_compute_data_proto, mesh_name),
            "collect_fn": partial(collect_lazy_compute_data_proto, mesh_name),
        }
```

然后实现 `dispatch_single_dp_lazy_compute_data_proto` 和 `collect_single_dp_lazy_compute_data_proto`。

#### 3.2 修改 `verl/workers/fsdp_workers.py`

修改 `generate_sequences` 的装饰器，指定目标 DP rank：

```python
# 方案 A: 硬编码特定 DP rank（不推荐）
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout", target_dp_rank=0))

# 方案 B: 通过配置传入（推荐）
# 需要在 Worker 初始化时保存 target_dp_rank，然后在装饰器中使用
```

### 方案三：在 WorkerGroup 层面过滤（最灵活）

在 WorkerGroup 中添加一个方法来创建只包含特定 DP rank workers 的子 WorkerGroup。

#### 3.1 修改 `verl/single_controller/base/worker_group.py`

添加方法：

```python
def filter_workers_by_dp_rank(self, mesh_name: str, target_dp_rank: int):
    """
    创建一个新的 WorkerGroup，只包含指定 DP rank 的 workers。
    
    Args:
        mesh_name: mesh 名称
        target_dp_rank: 目标 DP rank
        
    Returns:
        WorkerGroup: 新的 WorkerGroup 实例（只包含目标 workers）
    """
    # 查询 dispatch info
    if mesh_name not in self._dispatch_info:
        self._dispatch_info[mesh_name] = self._query_dispatch_info(mesh_name)
    
    dp_rank_mapping = self._dispatch_info[mesh_name]
    
    # 筛选出目标 DP rank 的 workers
    target_worker_indices = [
        i for i, dp_rank in enumerate(dp_rank_mapping) 
        if dp_rank == target_dp_rank
    ]
    
    # 创建新的 WorkerGroup（需要根据具体实现调整）
    filtered_workers = [self._workers[i] for i in target_worker_indices]
    filtered_worker_names = [self._worker_names[i] for i in target_worker_indices]
    
    # 创建新的 WorkerGroup 实例
    new_wg = type(self)(resource_pool=None)
    new_wg._workers = filtered_workers
    new_wg._worker_names = filtered_worker_names
    new_wg._dispatch_info = {mesh_name: [target_dp_rank] * len(filtered_workers)}
    
    return new_wg
```

#### 3.2 使用方式

```python
# 在调用 generate_sequences 之前
single_dp_wg = actor_rollout_wg.filter_workers_by_dp_rank("rollout", target_dp_rank=0)
output = single_dp_wg.generate_sequences(prompts)
```

## 四、推荐方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 方案一：新 Dispatch 模式 | 代码清晰，易于维护 | 需要修改多个文件，实现复杂 | 需要长期支持单 DP 域调用 |
| 方案二：修改现有函数 | 改动较小 | 可能影响现有功能 | 临时需求 |
| 方案三：WorkerGroup 过滤 | 最灵活，不影响现有代码 | 需要创建新的 WorkerGroup 实例 | 需要动态选择 DP 域 |

## 五、具体实现建议

### 推荐：方案三（WorkerGroup 过滤）+ 方案二的组合

1. **在 WorkerGroup 中添加过滤方法**（方案三）
2. **在调用时使用过滤后的 WorkerGroup**

这样可以：
- 保持现有代码不变
- 提供最大的灵活性
- 易于测试和调试

### 实现步骤

1. **修改 `verl/single_controller/base/worker_group.py`**
   - 添加 `filter_workers_by_dp_rank` 方法

2. **修改 `verl/single_controller/ray/base.py`**（如果是 RayWorkerGroup）
   - 实现 `filter_workers_by_dp_rank` 的具体逻辑
   - 确保新的 WorkerGroup 能正确绑定方法

3. **在使用处调用**
   ```python
   # 获取单个 DP 域的 WorkerGroup
   single_dp_wg = actor_rollout_wg.filter_workers_by_dp_rank("rollout", target_dp_rank=0)
   
   # 调用 generate_sequences
   output = single_dp_wg.generate_sequences(prompts)
   ```

## 六、注意事项

1. **TP 组完整性**：如果使用 TP（Tensor Parallel），需要确保整个 TP 组都在同一个 DP 域中，或者只选择完整的 TP 组。

2. **Collect Mask**：确保 `collect_mask` 正确设置，只有负责收集的 worker 才会返回结果。

3. **数据分片**：数据分片的大小需要根据实际参与计算的 workers 数量调整。

4. **错误处理**：如果目标 DP rank 不存在或没有足够的 workers，需要适当的错误处理。

5. **性能影响**：只使用部分 workers 可能会影响整体吞吐量，需要权衡。

## 七、测试建议

1. **单元测试**：测试 `filter_workers_by_dp_rank` 方法
2. **集成测试**：测试使用过滤后的 WorkerGroup 调用 `generate_sequences`
3. **性能测试**：对比使用全部 workers 和单个 DP 域的性能差异

