# Venue decision: MLSys 2027 Research Track

Decision date: 2026-09-08

## Decision

Target the **MLSys 2027 Research Track**. Do not submit this work concurrently
to another venue.

The paper's contribution is a causal systems-measurement instrument and a
prospective evidence sequence for asynchronous LLM reinforcement learning. The
MLSys call explicitly includes large-scale RL for LLMs, ML job scheduling,
training systems, and ML benchmarking/tooling. The research track is preferable
to the industrial track because the paper advances and validates a new
measurement method rather than primarily reporting a production deployment.

## Binding venue constraints

The official [MLSys 2027 Call for Research
Papers](https://mlsys.org/Conferences/2027/CallForResearchPapers) states:

- submission opens 2026-10-10 20:00 UTC;
- the paper and separate appendix are due 2026-10-30 20:00 UTC;
- the main paper is two-column and at most 10 pages, excluding references;
- references must explicitly list every author;
- appendices may be any length but reviewers are not required to read them;
- research-track submissions must make a good-faith effort to anonymize authors
  and institutions;
- dual submission is prohibited, while a preprint is permitted;
- artifact evaluation is voluntary and occurs after acceptance; and
- human authors must review and take responsibility for any LLM-assisted work.

MLSys 2027 reuses the official `mlsys2025` LaTeX style. The submission source in
`mlsys2027/` deliberately does not vendor an unchecked copy of that style; the
build instructions fetch or accept the exact official archive.

## Alternative considered

[ICLR 2027](https://iclr.cc/Conferences/2027/AuthorGuidelines) was considered
but not selected. It permits relevant systems and reinforcement-learning work,
but its nine-page submission limit and 2026-09-18 abstract/author-roster and
2026-09-25 full-paper deadlines leave materially less time for author review,
anonymization, and artifact preparation. The scientific framing is also more
directly aligned with MLSys's systems-measurement scope.

ICLR is not a chronological fallback after an MLSys decision: its deadline
precedes MLSys. A venue change before 2026-09-18 would therefore require an
explicit author decision, complete roster, and accelerated review. Nothing in
this package authorizes such a submission.

## Submission gates

Before upload, all of the following must be true:

1. Human authors approve the title, claims, and author list.
2. The compiled main paper is at most 10 pages before references.
3. The appendix is a separate PDF and no result needed for a primary claim
   appears only there.
4. The PDF, supplementary archive, links, and metadata pass an anonymity scan.
5. Every bibliography entry lists all authors and has been checked against a
   primary source.
6. Every numerical claim is traceable to a compact authenticated result record.
7. Artifact links are anonymous, non-tracking, and usable without internal
   credentials, or the paper clearly discloses the narrower artifact access.
8. No simultaneous archival submission is active.

## Scientific consequence

Venue selection does **not** create a need for another same-design M4
acquisition. The present evidence supports a scoped causal systems-measurement
paper. A downstream final-training-quality study would enlarge the claim, but
it is a separate hypothesis test and is not a prerequisite for the chosen
scope.
