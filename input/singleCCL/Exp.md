# Experiment Plan

2025/11/18

Yuxuan Li

## Objective

1. Validate the Simulation Framework
2. Packet Capture

## Validate the Simulation Framework

### Factors

* Topology
    - T1: Intra-Node Only (1x8)
    - T2: Inter-Node (Small-Scale, e.g., 2x1, 2x8)
    - T3: Inter-Node (Large-Scale, with multi-hop or congestion)
* Algorithm_XML
    - A1: AllReduce (Ring)
    - A2: AllReduce (Hierarchical)
    - A3: AllToAll
* Chunksize
    - C1: Latency-Bound
        - 1KB, 32KB, 64KB, 128KB, 256KB, 512KB
    - C2: Bandwidth-Bound
        - 1MB, 4MB, 16MB, 64MB, 256MB, 1GB
* Protocol
    - P1: Simple
    - P2: LL
    - P3: LL128
* Channel Number:
    - 1, 2, 4, 8

**<del>Totally 4 (Topology) x 3 (Algorithm_XML) x 12 (Chunksize) x 3 (Protocol) x 4 (Channel Number) = 1k+ configurations</del>**

### Baseline

- Topology = T2: Inter-Node (Small-Scale: 2x8)
- Algorithm = A1: AllReduce (Ring)
- Protocol = P1: Simple
- Channel Number = 2
- Chunksize = 64KB, 4MB, 256MB, 1GB

| **Topology** | Algorithm_XML       | Protocol | Channel Number | Chunksize | Latency (μs) |
|----------|---------------------|----------|-----------|-----------|---------------|
| T1       | A1: AllReduce (Ring) | P1: Simple | 2 | 64KB, 4MB, 256MB, 1GB | TBD |
| T2  2x1  | A1: AllReduce (Ring) | P1: Simple | 2 | 64KB, 4MB, 256MB, 1GB | TBD |
| T2  2x8  | A1: AllReduce (Ring) | P1: Simple | 2 | 64KB, 4MB, 256MB, 1GB | TBD |
| T3       | A1: AllReduce (Ring) | P1: Simple | 2 | 64KB, 4MB, 256MB, 1GB | TBD |


| Topology | **Algorithm_XML**       | Protocol | Channel Number | Chunksize | Latency (μs) |
|----------|---------------------|----------|-----------|-----------|---------------|
| T2  2x8  | A1: AllReduce (Ring) | P1: Simple | 2 | 64KB, 4MB, 256MB, 1GB | TBD |
| T2  2x8  | A2: AllReduce (Hierarchical) | P1: Simple | 2 | 64KB, 4MB, 256MB, 1GB | TBD |
| T2  2x8  | A3: AllToAll        | P1: Simple | 2 | 64KB, 4MB, 256MB, 1GB | TBD |

| Topology | Algorithm_XML       | **Protocol** | Channel Number | Chunksize | Latency (μs) |
|----------|---------------------|----------|-----------|-----------|---------------|
| T2  2x8  | A1: AllReduce (Ring) | P1: Simple | 2 | 64KB, 4MB, 256MB, 1GB | TBD |
| T2  2x8  | A1: AllReduce (Ring) | P2: LL | 2 | 64KB, 4MB, 256MB, 1GB | TBD |
| T2  2x8  | A1: AllReduce (Ring) | P3: LL128 | 2 | 64KB, 4MB, 256MB, 1GB | TBD |

| Topology | Algorithm_XML       | Protocol | **Channel Number** | Chunksize | Latency (μs) |
|----------|---------------------|----------|------------------|-----------|---------------|
| T2  2x8  | A1: AllReduce (Ring) | P1: Simple | 1 | 64KB, 4MB, 256MB, 1GB | TBD |
| T2  2x8  | A1: AllReduce (Ring) | P1: Simple | 2 | 64KB, 4MB, 256MB, 1GB | TBD |
| T2  2x8  | A1: AllReduce (Ring) | P1: Simple | 4 | 64KB, 4MB, 256MB, 1GB | TBD |
| T2  2x8  | A1: AllReduce (Ring) | P1: Simple | 8 | 64KB, 4MB, 256MB, 1GB | TBD |

| Topology | Algorithm_XML       | Protocol | Channel Number | **Chunksize** | Latency (μs) |
|----------|---------------------|----------|-----------|-----------|---------------|
| T2  2x8  | A1: AllReduce (Ring) | P1: Simple | 2 | 1KB, 32KB, 64KB, 128KB, 256KB, 512KB | TBD |
| T2  2x8  | A1: AllReduce (Ring) | P1: Simple | 2 | 1MB, 4MB, 16MB, 64MB, 256MB, 1GB | TBD |

**Totally 4 x (4 Topology + 3 Algorithm_XML + 3 Protocol + 4 Channel Number) + 12 (Chunksize) = 68 configurations**

### Tools

- perftest: link bandwidth
- msccl-test-nccl: algo bandwidth and [bus bandwidth](https://github.com/Azure/msccl-tests-nccl/blob/main/doc/PERFORMANCE.md#bandwidth)

### Procedure

TBD

20 warm-up runs, 100 measured runs


## Packet Capture

tcpdump-rdma
```bash
docker pull mellanox/tcpdump-rdma
docker run -it -v /dev/infiniband:/dev/infiniband -v /tmp/traces:/tmp/traces --net=host --privileged mellanox/tcpdump-rdma # bash Now start packet capture using RDMA device mlx5_0. (notice RDMA device, instead of Ethernet device)
tcpdump -i mlx5_0 -s 0 -w /tmp/traces/capture1.pcap #This will save packets in capture1.pcap file available in /tmp/traces directory inside and outside of the container.
```

