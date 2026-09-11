# Final post-PDF and reviewer-artifact adversarial review

Date: 2026-09-10

## Verdict

**Pass for bounded scientific review and local anonymous artifact delivery.**
The exact paper, supplement, and deterministic compact reviewer archive agree
on the six-cell Qwen evidence and the eight-acquisition Llama 3.2 1B/3B extensions.
They contain no discovered internal execution IDs, paths, credentials, or
author identity. No additional M4 GPU acquisition is needed
for the scoped opportunity-loss, bounded cross-family, and fixed-configuration
size-transport claims.

**Not yet a final conference-upload clearance.** Figure-derived fonts include
Type 3 and TrueType faces rather than the template's Type-1-only requirement;
the author roster and conflict metadata remain human gates; and no anonymous
public URL or empirical raw-ledger access is claimed.

## Exact objects reviewed

| Object | SHA-256 | Structure |
| --- | --- | --- |
| Main review PDF | `bb773c05c05b73507cf0074086044d3e163f7a728a464de57f4156c3988c67a4` | PDF 1.5, letter, 5 pages; references begin on page 4 |
| Supplement PDF | `05a7bfac27794afa092f60f57a597e9218bfd4d914ec7a6df84e46cb0e4e89e9` | PDF 1.5, letter, 2 pages |
| Reviewer archive | `8d6fd534d2375a0a2bd22a06639b6cd2b4547576333291775dd9280b62989158` | 26,077 bytes; 17 extracted files including manifest |
| Archive manifest | `a1784ac278a10a6b1bf6db4ae2f917ec6822c2d21c1732b562563a735f952bd1` | 16 hashed members plus A1--A14 external commitments |

## Adversarial checks

| Attack | Result | Evidence |
| --- | --- | --- |
| Headline-number drift | Pass | Verifiers reconcile 49,153 Qwen, 28,712 Llama 1B, and 28,788 Llama 3B assignments (106,653 total), zero missingness, Qwen correlations, and all Llama endpoints. |
| Shared-anchor double counting | Pass | Both interactions reuse the same two GSM8K draws; HAC and bootstrap covariance are retained and the artifact synthetic replay exercises shared-reference correlation. |
| Pipeline/scientific-status conflation | Pass | A3 remains `post-run fail` while its immutable completed acquisition is analyzed; no execution ID appears in the blind PDF or archive. |
| Retrospective promotion | Pass | The first acquisition remains `INCONCLUSIVE`; secondary synthesis and sensitivity analyses cannot override registered decisions. |
| Cross-family scope inflation | Pass | Llama is described as prospective replication in two fixed sizes and two workloads, not a randomized family contrast, pure architecture effect, or family-wide result. |
| Replicate cherry-picking | Pass | Both preregistered replicates enter every Llama size/workload endpoint with equal weight. The disclosed 3B replicate variation is not used to select a preferred run. |
| Llama workload overclaim | Pass | The OpenMath-minus-GSM8K contrast remains `INCONCLUSIVE`; the manuscript does not equate a null contrast with equality. |
| Llama size overclaim | Pass | Both 3B-minus-1B simultaneous intervals exclude zero, but size was fixed and only two points were tested; the paper prohibits causal and general scaling-law wording. |
| 3B joint-success inflation | Pass | OpenMath is `MATERIAL`, GSM8K is `INCONCLUSIVE`, and the manuscript explicitly records joint success as false. |
| PDF template leakage | Pass | Extracted text contains no `AUTHORERR`, suppressed-title message, private path, or internal execution identifier; metadata says `Anonymous Authors`. |
| Blind source linkage | Pass after repair | Searchable source revision prefixes were removed from the supplement; the private record retains the mapping. |
| Archive credentials/anonymity | Pass | Decompressed-content scan rejects internal host, repository, cluster, user, job, token, password, and known source-prefix patterns. |
| Archive traversal/symlink attack | Pass | Extraction validates every member stays under the temporary root and rejects symbolic and hard links. |
| Hidden project dependency | Pass | Replay, figure rendering, and two tests use Python 3.10+ standard library only under a four-variable credential-stripped environment. |
| Non-deterministic packaging | Pass | A second independent build is byte-identical to the reviewed archive. |
| Raw-data reproducibility overclaim | Pass | README and supplement state that empirical raw ledgers and public hosting are absent; the bundle verifies compact results and uses synthetic data only for end-to-end mechanics. |

## Repairs made during this review

1. Removed private implementation revision prefixes and the cluster name from
   the copied claim ledger.
2. Replaced a dangling private robustness filename with the bundle-local path
   and hash.
3. Added standalone figure-generation source and byte-for-byte SVG tests.
4. Removed an empty uncited bibliography from the supplement.
5. Removed source revision prefixes from the blind provenance table.
6. Added A7--A14 opaque commitments and deterministic checks for both Llama
   extensions without exposing pipeline or job identifiers.
7. Recomputed HAC/bootstrap, adjusted/unadjusted, missingness, threshold,
   common-window, and leave-one-replicate-out views from authenticated ledgers.
8. Added prespecified 3B-minus-1B contrasts, retained the failed co-primary
   joint decision, and rebuilt both PDFs with all-page visual inspection.

## Remaining release gates

1. Convert or outline figure text so final PDFs satisfy the Type-1-only font
   rule, then rerun font and all-page visual checks.
2. Supply the final author roster, affiliations, acknowledgments if any, and
   conflict metadata outside blind review mode.
3. Recover exact environment versions where authenticated records permit;
   otherwise retain the explicit omission.
4. Obtain approval before external hosting or raw-ledger release, then test the
   exact anonymous URL without internal credentials. The local archive itself
   is already credential-free and clean-room tested.
