from .worker_set import WorkerSet
from typing import Optional, Union, List, Dict

from reILs.infrastructure.data import Batch
from reILs.infrastructure.execution import WorkerSet, RolloutWorker

def local_sample(
    *,
    local_worker: RolloutWorker,
    max_steps: Optional[int] = None,
    concat: bool = False,
    render: bool = False,
) -> Union[List[Batch], Batch]:
    """
    Runs local rollouts on a local worker.
    Args:
        local_worker: The Local Worker to use for sampling.
        max_steps: Optional number of steps to be included in the sampled batch.
        concat: Whether to concat all resulting batches at the end and return the
            concated batch.
    Returns:
        The list of collected sample batch (one for each parallel
        rollout worker in the given `local_worker` if no max_steps).
    """
    steps = 0
    batch_list = []
    while (max_steps is None and steps == 0) or \
            (max_steps is not None and steps < max_steps):
        batches = [local_worker.sample()]
        # Update our counters for the stopping criterion of the while loop.
        for batch in batches:
            steps += len(batch)
        batch_list.extend(batches)

    if concat is True:
        batch_full = Batch.cat(batch_list)
        return batch_full
    else:
        return batch_list

def synchronous_parallel_sample(
    *,
    worker_set: WorkerSet,
    max_steps: Optional[int] = None,
    concat: bool = False
) -> Union[List[Batch], Batch]:
    """
    Runs parallel and synchronous rollouts on all remote workers.
    Waits for all workers to return from the remote calls.
    If no remote workers exist (num_workers == 0), use the local worker
    for sampling.

    Args:
        worker_set: The WorkerSet to use for sampling.
        max_steps: Optional number of steps to be included in the sampled batch.
        concat: Whether to concat all resulting batches at the end and return the
            concated batch.
    Returns:
        The list of collected sample batch (one for each parallel
        rollout worker in the given `worker_set` if no max_steps).
    """
    steps = 0
    batch_list = []
    while (max_steps is None and steps == 0) or \
            (max_steps is not None and steps < max_steps):
        # No remote workers in the set -> Use local worker for collecting
        # rollouts.
        if not worker_set.remote_workers():
            batches = [worker_set.local_worker().sample()]
        # Loop over remote workers' `sample()` method in parallel.
        else:
            batches = ray.get(
                [worker.sample.remote() for worker in worker_set.remote_workers()]
            )
        # Update our counters for the stopping criterion of the while loop.
        for batch in batches:
            steps += len(batch)
        batch_list.extend(batches)

    if concat is True:
        batch_full = Batch.cat(batch_list)
        return batch_full
    else:
        return batch_list
