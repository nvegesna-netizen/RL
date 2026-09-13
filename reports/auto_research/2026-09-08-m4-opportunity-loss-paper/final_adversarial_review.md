# Final post-PDF and reviewer-artifact adversarial review

Date: 2026-09-13

## Verdict

**Pass for bounded scientific review and local anonymous artifact delivery.**
The exact paper, supplement, and deterministic compact reviewer archive agree
on the six-cell Qwen evidence, the eight-acquisition Llama 3.2 1B/3B
extensions, and the separate 16-pair downstream-quality result.
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
| Main review PDF | `b148b6646c9af6fb1063211263ca08053c46496f66f11851b251fcb56d63475d` | PDF 1.5, letter, 6 pages; references begin on page 5 |
| Supplement PDF | `2916961f4295aad4ede8901a6d56d799f704411eb7b96630c91660949924aa69` | PDF 1.5, letter, 3 pages |
| Reviewer archive | `5e57d8b21d62cf39f1b27e598b20cf11b6609cc72d2c19267e7922d7ccdf3be6` | 27,625 bytes; 17 extracted files including manifest |
| Archive manifest | `95489b976864bab75d2ca355c7b663039df2c2126269372166568d1432776f53` | 16 hashed members, A1--A14 external commitments, and downstream hash chain |

## Adversarial checks

| Attack | Result | Evidence |
| --- | --- | --- |
| Headline-number drift | Pass | Verifiers reconcile 49,153 Qwen, 28,712 Llama 1B, and 28,788 Llama 3B assignments (106,653 total), zero missingness, Qwen correlations, all Llama endpoints, and the 16-block/32-run downstream result. |
| Shared-anchor double counting | Pass | Both interactions reuse the same two GSM8K draws; HAC and bootstrap covariance are retained and the artifact synthetic replay exercises shared-reference correlation. |
| Pipeline/scientific-status conflation | Pass | A3 remains `post-run fail` while its immutable completed acquisition is analyzed; no execution ID appears in the blind PDF or archive. |
| Retrospective promotion | Pass | The first acquisition remains `INCONCLUSIVE`; secondary synthesis and sensitivity analyses cannot override registered decisions. |
| Cross-family scope inflation | Pass | Llama is described as prospective replication in two fixed sizes and two workloads, not a randomized family contrast, pure architecture effect, or family-wide result. |
| Replicate cherry-picking | Pass | Both preregistered replicates enter every Llama size/workload endpoint with equal weight. The disclosed 3B replicate variation is not used to select a preferred run. |
| Llama workload overclaim | Pass | The OpenMath-minus-GSM8K contrast remains `INCONCLUSIVE`; the manuscript does not equate a null contrast with equality. |
| Llama size overclaim | Pass | Both 3B-minus-1B simultaneous intervals exclude zero, but size was fixed and only two points were tested; the paper prohibits causal and general scaling-law wording. |
| 3B joint-success inflation | Pass | OpenMath is `MATERIAL`, GSM8K is `INCONCLUSIVE`, and the manuscript explicitly records joint success as false. |
| Downstream-quality promotion | Pass | The paper reports -0.04492 with paired 95% CI [-0.13394, 0.04410] and labels it `INCONCLUSIVE`; it claims neither confirmed harm nor equivalence. |
| Downstream unit inflation | Pass | All inference uses 16 matched seed blocks, not 32 runs or 32,768 prompt scores as independent causal units. All 32 endpoints remain included. |
| Mediation overclaim | Pass | The downstream result is labeled a total release-policy effect; the paper does not identify M4 opportunity loss as the exclusive path to quality. |
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
9. Replaced the stale “final quality untested” statement with the prospective
   16-pair result, added the full interval and sensitivity boundary, and kept
   the proximal headline unchanged.
10. Added a public-safe downstream summary and six-record SHA-256 chain to the
    reviewer artifact, then passed credential-stripped replay and a byte-
    identical rebuild under Python 3.13, plus direct replay and tests under
    Python 3.10.

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
