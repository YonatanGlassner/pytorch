import sys
import threading

import torch
from torch import nn
from torch.autograd import Variable
from torch._utils import _flatten_tensors, _unflatten_tensors
from .collectives import all_reduce, get_num_processes

if sys.version_info[0] == 3:
    import queue
else:
    import Queue as queue


# import cupy
def _thread_fn():
    default_stream = torch.cuda.current_stream()

    # TODO: use per-device streams -- this assumes they're all on a single one
    def _process_batch():
        grad_batch, event, cuda_event = _reduction_queue.get()
        _reduction_stream.wait_event(cuda_event)
        coalesced = _flatten_tensors(grad_batch)
        coalesced *= (1. / get_num_processes())
        # cupy.cuda.nvtx.RangePush("all_reduce")
        all_reduce(coalesced)
        # cupy.cuda.nvtx.RangePop()
        for grad, reduced in zip(grad_batch, _unflatten_tensors(coalesced, grad_batch)):
            grad.copy_(reduced)
        event.set()

    with torch.cuda.stream(_reduction_stream):
        while True:
            _process_batch()  # just to have a clear scope


_reduction_queue = queue.Queue()
_reduction_stream = torch.cuda.Stream()
_reduction_thread = threading.Thread(target=_thread_fn, daemon=True)
_reduction_thread.start()


class DistributedDataParallel(nn.DataParallel):
    def __init__(self, *args, **kwargs):
        super(DistributedDataParallel, self).__init__(*args, **kwargs)
        self.bucket_bytes_cap = 1 * 1024 * 1024  # 1 MB

        self.bucket_sizes = []
        self.bucket_map = {}

        # Split parameters into buckets that will coalesce reductions
        # TODO: different types need different buckets
        bucket_bytes = self.bucket_bytes_cap  # to init the first bucket immediately
        for p in self.module.parameters():
            if bucket_bytes >= self.bucket_bytes_cap:
                self.bucket_sizes.append(0)
                bucket_bytes = 0
            self.bucket_sizes[-1] += 1
            self.bucket_map[p] = len(self.bucket_sizes) - 1
            bucket_bytes += p.numel() * p.element_size()

        self.buckets = [[] for _ in range(len(self.bucket_sizes))]
        self.reduced = [False] * len(self.bucket_sizes)

        self.dispatch_lock = threading.Lock()

        # Register callbacks on grad accumulators (post hooks)
        # TODO: this is unserializable
        self.grad_accs = []  # need to keep them in scope
        for p in self.module.parameters():
            p_tmp = p.expand_as(p)
            grad_acc = p_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(self._make_param_hook(p))
            self.grad_accs.append(grad_acc)

    def forward(self, *args, **kwargs):
        self.reduced = [False] * len(self.bucket_sizes)
        return super(DistributedDataParallel, self).forward(*args, **kwargs)

    def _make_param_hook(self, param):
        bucket_idx = self.bucket_map[param]
        def dist_dp_hook(*unused):
            if not param.grad.volatile:
                raise RuntimeError("DistributedDataParallel only works with volatile gradients")
            bucket = self.buckets[bucket_idx]
            bucket.append(param.grad.data)
            # This will be checked by _queue_reduction too, but it might
            # reduce contention on this lock
            if len(bucket) == self.bucket_sizes[bucket_idx]:
                with self.dispatch_lock:
                    self._queue_reduction(bucket_idx)
        return dist_dp_hook

    def _queue_reduction(self, bucket_idx):
        while bucket_idx >= 0:
            bucket = self.buckets[bucket_idx]
            # Check if it's ready
            if len(bucket) < self.bucket_sizes[bucket_idx]:
                return
            # Check that all buckets to the right have queued reductions
            is_last = bucket_idx == len(self.buckets) - 1
            if not is_last and not self.reduced[bucket_idx + 1]:
                return

            event = threading.Event()
            cuda_event = torch.cuda.Event()
            cuda_event.record()
            _reduction_queue.put((bucket, event, cuda_event))
            Variable._execution_engine.queue_callback(lambda: event.wait())
            if bucket_idx == 0:
                default_stream = torch.cuda.current_stream()
                Variable._execution_engine.queue_callback(
                    lambda: default_stream.wait_stream(_reduction_stream))
            self.buckets[bucket_idx] = []
            self.reduced[bucket_idx] = True

            # Try previous bucket
            bucket_idx -= 1
