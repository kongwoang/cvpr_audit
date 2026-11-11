"""Opt-in graph per existing fixed FP32 C1 context; no buffer/precision change."""
import torch

CHECKS = [0]


def install(encoder, counters, audit=False):
    assert encoder.kind == 'trt32'
    for runner in encoder.r.values():
        original = runner.run
        for _ in range(3):
            original()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            original()
        torch.cuda.synchronize()
        runner._latency_c1_graph = graph

        def run(runner=runner, original=original, graph=graph):
            if audit:
                original()
                reference = runner.out.clone()
            counters.EXEC[0] += 1
            graph.replay()
            if audit:
                assert torch.equal(reference, runner.out), 'C1 graph output mismatch'
                CHECKS[0] += 1

        runner.run = run
