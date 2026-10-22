# Lecture Tasks — 22 October 2026

Student: Abyz Nuradil. Student ID: **230103188** (taken from previous submissions in this repository).

Seven Java solutions and their **raw console outputs measured on the local Apple M2**, macOS 26.5.1, Java 20.0.2, 8 logical / 8 physical cores. No external libraries are required. Task 1 uses `double`, Java's IEEE 754 binary64 type, seeded with the Student ID.

## Reproduce

From this folder, with JDK 17 or newer:

```sh
mkdir -p build
javac -d build Bench.java Task*.java
for i in 1 2 3 4 5 6 7; do
  java -Xms1g -Xmx2g -cp build Task$i > task${i}_output.txt
done
```

Override the numeric seed/Student ID with `java -Dstudent.id=230103188 ...`. Task 3 detects physical cores through `sysctl` on macOS; elsewhere supply `-Dphysical.cores=N`. Timings and race outcomes vary between runs. The 2 GiB heap accommodates Task 7's two 240 MB arrays. Programs validate numerical results or completed counts; a failed validation terminates the run.

## Methods and interpretation

| Task | Implementation and timing scope |
|---|---|
| 1 | 20 million seeded doubles; serial sum and two-pass population standard deviation (`ddof=0`). Generation, computation and combined elapsed times printed separately. No worker pool. |
| 2 | Same seeded input; contiguous partitions and deterministic reduction with 1, 2, 4, 8, 16 workers. Median of 5 after 3 warmups. Dispatch, join and reduction included; pool creation and allocation excluded. T(1) times summation only, so is separate from Task 1. |
| 3 | 8 threads × 50 million increments; adjacent shared slots versus private accumulators with one final store. Opaque array accesses retain shared memory operations. Both loops read the same volatile increment step to keep the local loop observable. Start latch and joins delimit timing. Median of 3 with alternating order. |
| 4 | 1,000 prescribed periodic sleep costs; 4 workers. Static contiguous blocks versus an atomic-index dynamic queue, chunk 1. Three paired runs report total time, first/last completion and their delta. Uses dynamic scheduling, rather than a deque-based work-stealing implementation. |
| 5 | Prescribed six vector sizes, all 8 workers, reusable pool. Median of 9 paired batches after 5 warmups; 10 operations per batch, per-operation time reported. Dispatch/join included. Full output equality checked outside timing. Reports every observed slower-to-faster bracket, or explicitly states none. |
| 6 | 4 threads × 2 million increments. A volatile counter makes updates visible but `++` remains racy. Correct version uses AtomicInteger. Three runs in alternating order; synchronization time difference calculated from medians. |
| 7 | 30 million doubles; copy and prescribed trigonometric kernel, 1/2/8 workers. Median of 5 after 3 warmups. Full output checked outside timing. Effective copy bandwidth counts 16 bytes per element, excluding write allocation. |

The benchmark computations use real threads on this machine. Shared helper `Bench.java` supplies platform output, seed generation, timers and pool management. Java runtime/JIT/GC threads may also exist even for an application-level serial run. Pools are prestarted. Tasks 3 and 6 start timing at the release latch and include thread completion/join, while excluding thread construction and startup. Process startup is excluded throughout.

Amdahl's 95% assumption is a hypothetical model. Measured speedups can exceed that particular model if the actual serial fraction differs. Apple M2 has heterogeneous cores, so logical worker count does not imply equal compute capacity.

False sharing timing also includes repeated shared-store overhead and does not by itself isolate cache-line invalidations. The local variant avoids assuming a 64-byte hardware cache line.

Task 4's periodic costs are nearly balanced even with static blocks; dynamic scheduling need not produce a large total-time improvement. OS sleep granularity also affects results.

Task 5 found **no slower-to-faster crossover in the prescribed sizes on this run**. Reporting a fabricated threshold would misrepresent the measurement. A measured bracket, when one exists on a rerun, cannot locate an exact threshold among untested sizes.

Task 6's atomic version lost zero updates and was faster than the contended volatile read-modify-write version. Synchronization therefore had a negative measured time penalty in this experiment; a positive penalty is not assumed.

For Task 7, relative scaling is evidence consistent with a bandwidth bottleneck, not proof of memory-channel saturation. The program reports its inference based on the measured efficiencies and distinguishes inconclusive measurements. Hardware performance counters would be needed to attribute a stall precisely.
