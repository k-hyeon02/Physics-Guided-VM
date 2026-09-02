"""CPU 샘플 준비와 단일 gpuRIR 렌더러를 분리한 스트리밍 로더."""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from .dataset import ChannelGroupBatchSampler, SyntheticDOADataset


@dataclass(frozen=True)
class _RemoteError:
    """자식 프로세스의 예외를 메인 프로세스로 전달하기 위한 자료구조."""

    stage: str
    message: str
    traceback_text: str
    sample_index: int | None = None


class StreamingPipelineError(RuntimeError):
    """CPU 준비 또는 gpuRIR 렌더러에서 발생한 원격 오류."""


def _remote_error(
    stage: str,
    error: BaseException,
    sample_index: int | None = None,
) -> _RemoteError:
    return _RemoteError(
        stage=stage,
        message=f"{type(error).__name__}: {error}",
        traceback_text=traceback.format_exc(),
        sample_index=sample_index,
    )


def _prepare_worker(
    worker_id: int,
    dataset: SyntheticDOADataset,
    task_queue,
    prepared_queue,
) -> None:
    """CUDA를 사용하지 않고 ``SimulationRequest``만 생성."""

    active_state: tuple[int, str] | None = None
    sample_index: int | None = None
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break

            batch_id, position, sample_index, epoch, profile = task
            requested_state = (epoch, profile)
            if active_state != requested_state:
                dataset.set_epoch(epoch)
                dataset.set_profile(profile)
                active_state = requested_state

            request = dataset.prepare_sample(sample_index)
            prepared_queue.put(
                ("sample", batch_id, position, sample_index, request)
            )
            sample_index = None
    except BaseException as error:
        prepared_queue.put(
            ("error", _remote_error("prepare", error, sample_index))
        )
    finally:
        prepared_queue.put(("worker_done", worker_id))


def _render_worker(
    dataset: SyntheticDOADataset,
    prepared_queue,
    output_queue,
    batch_sizes: Sequence[int],
    num_prepare_workers: int,
) -> None:
    """요청을 원래 순서로 렌더링하고 완성된 batch를 미리 전송."""

    pending: dict[tuple[int, int], tuple[int, object]] = {}
    current_samples: list[dict[str, torch.Tensor]] = []
    next_batch = 0
    next_position = 0
    finished_workers = 0
    sample_index: int | None = None

    try:
        while next_batch < len(batch_sizes):
            message = prepared_queue.get()
            kind = message[0]

            if kind == "error":
                output_queue.put(message)
                return
            if kind == "worker_done":
                finished_workers += 1
            elif kind == "sample":
                _, batch_id, position, index, request = message
                pending[(batch_id, position)] = (index, request)
            else:
                raise RuntimeError(f"알 수 없는 prepared message: {kind!r}")

            expected = (next_batch, next_position)
            while expected in pending:
                sample_index, request = pending.pop(expected)
                current_samples.append(dataset.render_sample(request))
                sample_index = None
                next_position += 1

                if next_position == batch_sizes[next_batch]:
                    tensor_batch = default_collate(current_samples)
                    output_queue.put(
                        (
                            "batch",
                            next_batch,
                            {
                                key: value.detach().cpu().numpy()
                                for key, value in tensor_batch.items()
                            },
                        )
                    )
                    current_samples = []
                    next_batch += 1
                    next_position = 0
                    if next_batch == len(batch_sizes):
                        return

                expected = (next_batch, next_position)

            if finished_workers == num_prepare_workers and expected not in pending:
                raise RuntimeError(
                    "모든 CPU preparation worker가 종료됐지만 "
                    f"batch={next_batch}, position={next_position} 요청이 누락됨"
                )
    except BaseException as error:
        output_queue.put(
            ("error", _remote_error("render", error, sample_index))
        )


def _numpy_batch_to_tensors(
    batch: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(value) for key, value in batch.items()}


def _pin_batch_memory(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.pin_memory() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


class StreamingSimulationLoader:
    """DataLoader와 유사한 iterator로 CPU 준비와 gpuRIR 렌더링을 중첩.

    worker와 renderer는 iterator마다 새로 생성한다. gpuRIR/cuFFT 상태가 epoch
    사이에 누적되지 않아 기존 ``persistent_workers=False``의 격리 특성을
    유지한다. 큰 배열은 ``prefetch_batches + 1``개 batch 범위 안에서만
    준비되므로 메모리 사용량도 제한된다.
    """

    def __init__(
        self,
        dataset: SyntheticDOADataset,
        batch_sampler,
        num_prepare_workers: int,
        prefetch_batches: int = 2,
        pin_memory: bool = True,
        multiprocessing_context: str = "spawn",
        queue_poll_seconds: float = 1.0,
        shutdown_timeout_seconds: float = 2.0,
    ) -> None:
        if num_prepare_workers < 1:
            raise ValueError("num_prepare_workers는 1 이상이어야 함")
        if prefetch_batches < 1:
            raise ValueError("prefetch_batches는 1 이상이어야 함")
        if queue_poll_seconds <= 0.0:
            raise ValueError("queue_poll_seconds는 0보다 커야 함")
        if shutdown_timeout_seconds < 0.0:
            raise ValueError("shutdown_timeout_seconds는 음수가 될 수 없음")

        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.num_prepare_workers = int(num_prepare_workers)
        self.prefetch_batches = int(prefetch_batches)
        self.pin_memory = bool(pin_memory)
        self.multiprocessing_context = multiprocessing_context
        self.queue_poll_seconds = float(queue_poll_seconds)
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self._iteration_active = False

    def __len__(self) -> int:
        return len(self.batch_sampler)

    @staticmethod
    def _format_remote_error(error: _RemoteError) -> str:
        location = (
            ""
            if error.sample_index is None
            else f" (sample index {error.sample_index})"
        )
        return (
            f"streaming {error.stage} worker failed{location}: {error.message}\n"
            f"{error.traceback_text}"
        )

    def _shutdown(self, processes: Sequence[mp.Process], queues: Sequence) -> None:
        for process in processes:
            process.join(timeout=self.shutdown_timeout_seconds)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=self.shutdown_timeout_seconds)

        for process_queue in queues:
            try:
                process_queue.cancel_join_thread()
            except (AttributeError, ValueError):
                pass
            try:
                process_queue.close()
            except (AttributeError, ValueError):
                pass

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self._iteration_active:
            raise RuntimeError("같은 StreamingSimulationLoader를 동시에 순회할 수 없음")

        batches = [list(map(int, batch)) for batch in self.batch_sampler]
        if not batches:
            return

        self._iteration_active = True
        context = mp.get_context(self.multiprocessing_context)
        task_queue = context.Queue()
        pipeline_depth = min(len(batches), self.prefetch_batches + 1)
        max_batch_size = max(map(len, batches))
        prepared_queue = context.Queue(
            maxsize=max(
                self.num_prepare_workers,
                pipeline_depth * max_batch_size,
            )
        )
        output_queue = context.Queue(maxsize=self.prefetch_batches)

        epoch = int(self.dataset._epoch)
        profile = str(self.dataset.profile)
        batch_sizes = [len(batch) for batch in batches]

        prepare_processes = [
            context.Process(
                target=_prepare_worker,
                args=(worker_id, self.dataset, task_queue, prepared_queue),
                name=f"simulation-prepare-{worker_id}",
                daemon=True,
            )
            for worker_id in range(self.num_prepare_workers)
        ]
        render_process = context.Process(
            target=_render_worker,
            args=(
                self.dataset,
                prepared_queue,
                output_queue,
                batch_sizes,
                self.num_prepare_workers,
            ),
            name="simulation-render",
            daemon=True,
        )
        processes = [*prepare_processes, render_process]

        next_batch_to_enqueue = 0
        sentinels_sent = False

        def enqueue_batch(batch_id: int) -> None:
            for position, sample_index in enumerate(batches[batch_id]):
                task_queue.put(
                    (batch_id, position, sample_index, epoch, profile)
                )

        def enqueue_until_window_full() -> None:
            nonlocal next_batch_to_enqueue, sentinels_sent
            target = min(
                len(batches),
                next_batch_to_enqueue + pipeline_depth,
            )
            while next_batch_to_enqueue < target:
                enqueue_batch(next_batch_to_enqueue)
                next_batch_to_enqueue += 1
            if next_batch_to_enqueue == len(batches) and not sentinels_sent:
                for _ in prepare_processes:
                    task_queue.put(None)
                sentinels_sent = True

        try:
            for process in processes:
                process.start()
            enqueue_until_window_full()

            expected_batch = 0
            while expected_batch < len(batches):
                try:
                    message = output_queue.get(timeout=self.queue_poll_seconds)
                except queue.Empty:
                    failed = [
                        process
                        for process in processes
                        if process.exitcode not in (None, 0)
                    ]
                    if failed:
                        details = ", ".join(
                            f"{process.name} exit={process.exitcode}"
                            for process in failed
                        )
                        raise StreamingPipelineError(
                            f"streaming worker가 예고 없이 종료됨: {details}"
                        )
                    if not render_process.is_alive():
                        raise StreamingPipelineError(
                            "simulation renderer가 batch를 반환하기 전에 종료됨"
                        )
                    continue

                kind = message[0]
                if kind == "error":
                    raise StreamingPipelineError(
                        self._format_remote_error(message[1])
                    )
                if kind != "batch":
                    raise StreamingPipelineError(
                        f"알 수 없는 output message: {kind!r}"
                    )

                _, batch_id, numpy_batch = message
                if batch_id != expected_batch:
                    raise StreamingPipelineError(
                        f"batch 순서 오류: expected={expected_batch}, got={batch_id}"
                    )

                expected_batch += 1
                if next_batch_to_enqueue < len(batches):
                    enqueue_batch(next_batch_to_enqueue)
                    next_batch_to_enqueue += 1
                    if (
                        next_batch_to_enqueue == len(batches)
                        and not sentinels_sent
                    ):
                        for _ in prepare_processes:
                            task_queue.put(None)
                        sentinels_sent = True

                batch = _numpy_batch_to_tensors(numpy_batch)
                if self.pin_memory and torch.cuda.is_available():
                    batch = _pin_batch_memory(batch)
                yield batch
        finally:
            self._shutdown(
                processes,
                (task_queue, prepared_queue, output_queue),
            )
            self._iteration_active = False


def build_streaming_dataloader(
    dataset: SyntheticDOADataset,
    batch_size: int,
    num_prepare_workers: int,
    shuffle: bool,
    prefetch_batches: int = 2,
    drop_last: bool = True,
    pin_memory: bool = True,
    multiprocessing_context: str = "spawn",
) -> StreamingSimulationLoader:
    """채널 수 그룹화를 적용한 streaming simulation loader 생성."""

    sampler = ChannelGroupBatchSampler(
        channel_counts=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return StreamingSimulationLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_prepare_workers=num_prepare_workers,
        prefetch_batches=prefetch_batches,
        pin_memory=pin_memory,
        multiprocessing_context=multiprocessing_context,
    )
