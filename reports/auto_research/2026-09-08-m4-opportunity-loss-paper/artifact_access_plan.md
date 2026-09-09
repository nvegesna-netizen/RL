# Anonymous artifact-access plan

Status: **planned, not released**

No external upload, repository publication, or access-policy change is
authorized by this document.

## Reviewer bundle

Create one anonymous, credential-free archive containing:

- the manuscript's compact result JSON files and claim ledger;
- analysis and figure-generation source plus unit tests;
- frozen protocol JSON files and public-safe configuration overlays;
- a machine-readable provenance manifest with SHA-256 and byte size for every
  included file and every external raw artifact;
- a synthetic miniature ledger that exercises the complete analysis path; and
- exact offline commands and expected output hashes.

The reviewer bundle must not contain internal hostnames, GitLab URLs, pipeline
URLs, employee names, usernames, home paths, job comments, credentials,
environment dumps, or model/dataset tokens. Public-safe opaque acquisition IDs
should replace internal pipeline and job identifiers in the anonymous bundle;
the private provenance ledger retains the reversible mapping.

## Raw terminal artifacts

The six terminal archives total hundreds of megabytes and remain outside Git.
Preferred access is an anonymous, read-only object store with stable URLs,
checksums, content length, and no request tracking exposed to authors during
review. If policy permits, publish the lifecycle and opportunity ledgers after
scrubbing log/environment material; the full raw logs are not necessary for
recomputing the estimands.

If anonymous public hosting is not approved before submission, ship compact
records and synthetic data, keep all raw SHA-256 commitments in the paper, and
state plainly that authenticated raw ledgers are available to the artifact
evaluation committee under controlled access. Do not imply public availability.

## Release gates

1. Legal/data-owner approval for each dataset-derived and log-derived field.
2. Secret scan and manual privacy review of decompressed content.
3. Anonymity scan for organization, repository, pipeline, cluster, and user
   identifiers.
4. Clean-room download on a machine without NVIDIA-internal credentials.
5. Hash verification and complete offline analysis replay.
6. Link check from the exact submission PDF and supplementary README.
7. Human-author approval of the final disclosure and access limitations.

## Camera-ready path

After deanonymization and acceptance, replace opaque IDs with the canonical
public repository revision, publish the scrubbed raw-ledger bundle if approved,
and apply for MLSys artifact evaluation. Preserve the submitted anonymous
bundle's checksum so reviewers' evaluated object remains identifiable.
