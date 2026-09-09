# Final post-PDF and reviewer-artifact adversarial review

Date: 2026-09-08

## Verdict

**Pass for bounded scientific review and local anonymous artifact delivery.**
The exact paper, supplement, and deterministic compact reviewer archive agree
on the six-cell evidence and contain no discovered internal execution IDs,
paths, credentials, disclosure text, or author identity. No additional M4 GPU
acquisition is needed for the scoped opportunity-loss claim.

**Not yet a final conference-upload clearance.** Figure-derived fonts include
Type 3 and TrueType faces rather than the template's Type-1-only requirement;
the author roster and conflict metadata remain human gates; and no anonymous
public URL or empirical raw-ledger access is claimed.

## Exact objects reviewed

| Object | SHA-256 | Structure |
| --- | --- | --- |
| Main review PDF | `e79f36bd1c83063e8a04a7d08c13d3730757abc1d7d28e3ead37185b96c8d7e4` | PDF 1.5, letter, 5 pages; references begin on page 4 |
| Supplement PDF | `8c7931300784b25627d42dc3339432b3ee479b1862dfaab31b5e40bbc1dcd499` | PDF 1.5, letter, 2 pages |
| Reviewer archive | `eada56b991c220ac6726eef56fc458544009364a87b570a50f9efce3602091d2` | 18,289 bytes; 17 extracted files including manifest |
| Archive manifest | `4995abb4e60c5c504836b65e27cadf66573cea4821d698530e560fc1f2392ecd` | 16 hashed members plus A1--A6 external commitments |

## Adversarial checks

| Attack | Result | Evidence |
| --- | --- | --- |
| Headline-number drift | Pass | Publication verifier reconciles 49,153 full-window and 43,756 common-window assignments, zero missingness, correlation values, and synthesis hash. |
| Shared-anchor double counting | Pass | Both interactions reuse the same two GSM8K draws; HAC and bootstrap covariance are retained and the artifact synthetic replay exercises shared-reference correlation. |
| Pipeline/scientific-status conflation | Pass | A3 remains `post-run fail` while its immutable completed acquisition is analyzed; no execution ID appears in the blind PDF or archive. |
| Retrospective promotion | Pass | The first acquisition remains `INCONCLUSIVE`; secondary synthesis and sensitivity analyses cannot override registered decisions. |
| Scope inflation | Pass | The claim remains limited to tested Qwen3 GRPO math-workload environments and excludes final model quality and family-wide generalization. |
| PDF template leakage | Pass | Extracted text contains no `AUTHORERR`, suppressed-title message, disclosure, private path, or internal execution identifier; metadata says `Anonymous Authors`. |
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
