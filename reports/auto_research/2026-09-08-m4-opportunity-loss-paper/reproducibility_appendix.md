# Reproducibility and provenance appendix

This appendix distinguishes facts authenticated by preserved records from
details that were not captured. It intentionally does not infer GPU memory,
driver version, CUDA version, or host CPU from the EOS resource name.

## Hardware and execution environment

| Field | Evidence-backed value |
| --- | --- |
| Accelerator allocation | Two `dgxh100_eos` H100-class accelerator slots per acquisition |
| Visible devices | `CUDA_VISIBLE_DEVICES=0,1` in the acquisition manifests |
| Execution environment | EOS, one generated two-GPU workload per cell |
| Training backend | NeMo RL single-controller GRPO with Megatron policy workers |
| Generation backend | asynchronous vLLM generation worker |
| Container | `nemo-rl-nightly-5802754.sqsh` |
| Container SHA-256 | `3df8114a0b3e60ef95ce13f8b982c7cc45d164aca71c63a87388b1e8434ee470` |
| Container embedded NeMo RL commit | `ae07eafe8035b5b2e84efa7234e70e7fd7e493c1` |
| Python runtime evidence | Python 3.13 paths appear in preserved worker logs |

The frozen source archive for each acquisition was overlaid on the pinned
container. Therefore, the per-cell source commit in the provenance table—not
only the container's embedded commit—defines the implementation under test.
Exact driver, CUDA, PyTorch, vLLM, Megatron, CPU, memory-capacity, and operating-
system versions were not written into the compact terminal records. They must
be recovered from authenticated logs before camera-ready publication or listed
as unavailable; they must not be reconstructed from memory.

## Shared experimental contract

The assignment unit is an epoch-specific prompt group. Each group contains
eight sibling generations. Opportunity is measured before release. The two
definitive arms are immediate release (`control`) and an added five-second hold
(`d5`). The learner uses a global training batch size of 32, a rollout buffer
capacity of 64, 16 in-flight prompts, windowed FIFO sampling, and a maximum
staleness of one version in the accepted OpenMath design; transport protocols
freeze their deviations explicitly.

The primary adjusted estimator uses eight contiguous start-version folds and
only pre-treatment opportunity and its zero indicator as nuisance features.
Dependence is indexed by start version. The registered uncertainty procedures
are a four-lag Newey--West HAC interval and a 20,000-draw circular-block
bootstrap with block length eight versions. The reported outer envelope is the
union of the two intervals across sharp terminal-missingness endpoints.

No terminal assignment is missing in any definitive full-window acquisition.
Qualification observations and failed-attempt observations do not enter any
primary estimator. No automatic extension or outcome-guided retry was allowed.

## Definitive cell provenance

`Pipeline` below is the downstream pipeline that executed the acquisition.
`Artifact SHA-256` authenticates the preserved terminal archive, not merely a
compact result JSON.

| Cell | Pipeline | Pipeline state | Source commit | Protocol SHA-256 | Artifact SHA-256 |
| --- | ---: | --- | --- | --- | --- |
| Qwen3-0.6B / OpenMath | 65986159 | success | `47dc1a713cbf0e07813d31c4ba70a8262214e098` | `6631a4ca8cfc6010150d1c135a1f5821c717f0dea189a3309c5f0f58f9b0709c` | `07cddc489ea20055161f9c435d44d0a52b91113fda9eac16472bdb531dc6ffb6` |
| Qwen3-1.7B / OpenMath | 66186620 | success | `a1c719177704b7d14cf1d503dc96f0f2f9f1701d` | `eef87a5e26f2cc1a39428628b0c28288b17a297f49d449a80ed9699f9ee17cb7` | `022e4d18f2eedd2339c3c7eb7decf6e21f929761de46bcd0f85a73f0a2700ac7` |
| Qwen3-0.6B / GSM8K | 66675506 | failed after training, at analysis invocation | `bfcf8447ac2395520a4fc3805859f167fcbb2bb0` | `b6682795071d12a5d6b3eddcdf73536e045949f08ce881885dd330fba9885db6` | `6c2ebeb2b2bf67df16be815f372ea87cde8eeff4d23e63291d8a98ac94659075` |
| Qwen3-1.7B / GSM8K | 66571072 | success | `b79e419aea95b5226ac4274c9a1535251658b5a5` | `3413bde3718563157cee5d504f174710f65406dd7f1effd8ad2755a03d73b69e` | `966dfbf60548e5fd791d59d3baa2bc3b3a1b8d6b0b1a15de1ef510fc618720ca` |
| Qwen3-0.6B / NuminaMath | 66843373 | success | `02b0ad4abfedc80994f71f8a550473431154b823` | `bbe2c07211952e3e46d4511524f68e864606a7c63f271ae8e064263044dad0c8` | `ad7c556f435b0f9e30a65c201680bf6dab74e077af16e38f299bfc04af969c35` |
| Qwen3-1.7B / NuminaMath | 66843327 | success | `02b0ad4abfedc80994f71f8a550473431154b823` | `bbe2c07211952e3e46d4511524f68e864606a7c63f271ae8e064263044dad0c8` | `6aa61968730833387a15b45a6d0f2be541c6f84b6d4f16a29e7dbf583981fc9b` |
| Llama 3.2 1B / OpenMath r1 | 67204578 | success | `8f5151cf00e02933a9b92772443d62e95b5150ff` | `56a3311f78a0ac12cb9dc7ecdeee1a6fc0c5ace889799ebfaba1faaa0e610521` | `7fc1d62dba8bf82777e9d86abc3e205afa5f5012ce067082fd3a2babe88139b6` |
| Llama 3.2 1B / OpenMath r2 | 67204614 | success | `8f5151cf00e02933a9b92772443d62e95b5150ff` | `56a3311f78a0ac12cb9dc7ecdeee1a6fc0c5ace889799ebfaba1faaa0e610521` | `37da83c526bef6cf13b36090d21dd49a9d73a768422e3fea9381521eb8e71a33` |
| Llama 3.2 1B / GSM8K r1 | 67204678 | success | `8f5151cf00e02933a9b92772443d62e95b5150ff` | `56a3311f78a0ac12cb9dc7ecdeee1a6fc0c5ace889799ebfaba1faaa0e610521` | `c321dcf94a138d2a289142b0991c452be848d28dedf5f5c468fba78cd85ffa59` |
| Llama 3.2 1B / GSM8K r2 | 67204582 | success | `8f5151cf00e02933a9b92772443d62e95b5150ff` | `56a3311f78a0ac12cb9dc7ecdeee1a6fc0c5ace889799ebfaba1faaa0e610521` | `fcb236655ed1554215e370cca6d21fe5873b385e4f4b1576cdc27ff2dbe4735b` |
| Llama 3.2 3B / OpenMath r1 | 67263519 | success | `8f5151cf00e02933a9b92772443d62e95b5150ff` | `4f2a5e496030e836b3449697928f4f363d7d36e3cec668105b2cdcc701b9814c` | `b79b7d1bcd861bdacf3d161bbdc5fe236edc35da41f427d0adf20a1895ddcd38` |
| Llama 3.2 3B / OpenMath r2 | 67263508 | success | `8f5151cf00e02933a9b92772443d62e95b5150ff` | `4f2a5e496030e836b3449697928f4f363d7d36e3cec668105b2cdcc701b9814c` | `034d6e57440593a48a28f705534474b1c36c8aeb0170c22e54d869f4e33a9aaa` |
| Llama 3.2 3B / GSM8K r1 | 67263541 | success | `8f5151cf00e02933a9b92772443d62e95b5150ff` | `4f2a5e496030e836b3449697928f4f363d7d36e3cec668105b2cdcc701b9814c` | `6714f77466a5d3e3c854339f69d19ac4d8312276cc3627c25fe3d8ef79b26163` |
| Llama 3.2 3B / GSM8K r2 | 67263552 | success | `8f5151cf00e02933a9b92772443d62e95b5150ff` | `4f2a5e496030e836b3449697928f4f363d7d36e3cec668105b2cdcc701b9814c` | `e535f0ce6c1fdb4f1863eca4142596f4543190d53dfd70cee7ddaf35d25a5920` |

Pipeline 66675506 is deliberately not labeled successful. It completed all 558
registered trainer steps and preserved 9,573 scored primary assignments, then
called an analyzer that rejected the grid protocol identity. The preregistered
correct analyzer was run on the immutable ledgers without new acquisition or
protocol change. Its result SHA-256 is
`26d1650143c502e5790f194ab7c703051820e47993c8fcd9206bbec2bd7f10c2`.
This is scientific acquisition completion with a post-run packaging failure,
not pipeline success.

## Historical instrument and first acquisition

The accepted instrument was frozen at source commit
`9cc2e9c6e339f7cd3c0e83fa9abf80af9cf428a6`. The first 224-step acquisition
ran in downstream pipeline 65912780 under protocol
`25ca26c07ebc6c52b3b67447803b3a73e0eca4306a6b5816c954415b0638acad`.
Its terminal archive SHA-256 is
`e81a29747628b6f0490323207f317755de411d0f8e7527f715368919aee12fb7`.
The parent pipeline failed during packaging after the canonical result existed.
The scientific decision remains `INCONCLUSIVE`.

## Prospective downstream-quality provenance

The end-to-end study used 16 matched training-seed blocks and 32 independently
trained Qwen3-0.6B/OpenMath runs. Within each block, one run used all-immediate
release and one used an equal-mass control/d5 policy. Every run used the frozen
448-update budget and the same ordered 1,024-prompt terminal evaluation. The
training-seed block—not the prompt—is the causal unit.

| Record | SHA-256 |
| --- | --- |
| Protocol | `dbe7a4f7d1938ef43d24e536ccf9dd57110a01bab5e6cb511980218e472ef2c0` |
| 32-run manifest | `a461e83c7dbccdefa1c5c779062e709f34bfc502639a3e7de663a717fefa93bf` |
| Terminal topology | `b644466687e59f89d4ddb33d75f3b1c096db9e10f7090b122d263a4b2068d72c` |
| Terminal authentication | `f1c00b24877ba54fc68ff267af86824aa2233f8bdf674416bfaa18c9a801a441` |
| Completion gate | `54167b9517dc37b3ae05122e035c95da5bcd50f75d2326c1d6a841c7ddc43887` |
| Frozen analysis | `b412c5bb61ae637bf8e52442df09b8fec8e21800123ed2d900b987feca6da306` |

All 32 workload ZIPs passed integrity checks; their signed manifests and all
eight registered members matched declared hashes. The completion gate was
`COMPLETE_AUTHENTICATED_16_PAIRS`. Eighteen runs use their original successful
workload artifacts; the 14 b10--b16 identities use the prospectively authorized
runtime-recovery-v2 artifacts. Every selected scientific workload succeeded.
The b02 immediate parent pipeline remains failed solely because its logs-after
transfer failed; its workload artifact and endpoint authenticated and were
retained. Measured use was 52.1158 wall-hours and 104.2317 H100 GPU-hours,
below the frozen 128-wall-hour and 256-H100-hour caps.

The registered mixed-d5-minus-immediate estimate is -0.04492 with paired
Student 95% interval [-0.13394, 0.04410]. The exact 65,536-assignment sign-flip
test gives p=0.29816; paired-bootstrap and covariate-adjusted sensitivities do
not change the `INCONCLUSIVE` decision. No run, including six zero-accuracy
endpoints, was excluded.

## Assignment and observer-duty audit

| Cell | Full-window assignments | Unscored | Corrected observer duty | Mechanism |
| --- | ---: | ---: | ---: | --- |
| Qwen3-0.6B / OpenMath | 7,199 | 0 | 0.002174 | replicated |
| Qwen3-1.7B / OpenMath | 8,673 | 0 | 0.001853 | replicated |
| Qwen3-0.6B / GSM8K | 9,573 | 0 | 0.003286 | replicated |
| Qwen3-1.7B / GSM8K | 9,429 | 0 | 0.002943 | replicated |
| Qwen3-0.6B / NuminaMath | 7,229 | 0 | 0.002121 | replicated |
| Qwen3-1.7B / NuminaMath | 7,050 | 0 | 0.001798 | replicated |
| Llama 3.2 1B / OpenMath r1 | 6,908 | 0 | 0.000656 | replicated |
| Llama 3.2 1B / OpenMath r2 | 6,809 | 0 | 0.000603 | replicated |
| Llama 3.2 1B / GSM8K r1 | 7,314 | 0 | 0.000731 | replicated |
| Llama 3.2 1B / GSM8K r2 | 7,681 | 0 | 0.000782 | replicated |
| Llama 3.2 3B / OpenMath r1 | 7,073 | 0 | 0.000436 | replicated |
| Llama 3.2 3B / OpenMath r2 | 6,612 | 0 | 0.000417 | replicated |
| Llama 3.2 3B / GSM8K r1 | 7,176 | 0 | 0.000547 | replicated |
| Llama 3.2 3B / GSM8K r2 | 7,927 | 0 | 0.000513 | replicated |

Every value is below the frozen 0.01 observer-duty ceiling. This supports the
registered instrumentation-overhead qualifier within these runs; it is not a
hardware portability claim.

## Dataset authentication

GSM8K uses `openai/gsm8k`, `main`, training split. NuminaMath uses the frozen
NuminaMath-1.5 revision
`1b05109f9e5c1ad06c0663519502416c30b300f8`; 896,215 source rows were filtered
to 680,786 unique retained problems. The three parquet shard hashes and sizes
are preserved in the NuminaMath preflight lock. OpenMath provenance remains
bound through its protocol and terminal artifact; a public-facing dataset
revision must be recovered and stated before release if it is absent from that
protocol.

## Reproduction levels

1. **Compact-result verification:** validate JSON schemas and SHA-256 links,
   regenerate tables and figures. This requires no GPU.
2. **Raw-ledger reanalysis:** authenticate lifecycle/opportunity ledgers and
   rerun all cell, interaction, and robustness analyses. This requires the
   external artifact bundle but no model training.
3. **Acquisition replication:** rebuild the pinned software environment and run
   the registered two-GPU protocol. This requires model/dataset access and EOS-
   equivalent accelerator resources.

The repository currently supports level 1 and internal level 2. Public reviewer
access remains gated by the artifact plan.
