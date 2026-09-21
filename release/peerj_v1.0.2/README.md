# PeerJ v1.0.2 local release staging

This directory stages the deterministic source and reproducibility archive for
the documentation-only Mamba-TransQR v1.0.2 PeerJ synchronization patch.

The ZIP payload is intentionally excluded from Git history. `SHA256SUMS.txt`
records the final archive and publication-stable README hashes. The archive
contains the v1.0.2 source tree under `Mamba-TransQR-1.0.2/` and excludes this
staging directory and the prepublication audit files to avoid checksum
self-reference.

This staging area contains no dataset, checkpoint, benchmark-output, M3D, or
mobile bundle. Scientific artifacts remain associated with the frozen v1.0.0
release and are unchanged by v1.0.2.
