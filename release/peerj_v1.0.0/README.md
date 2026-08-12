# PeerJ v1.0.0 local archival staging

This directory stages future Zenodo assets; nothing here has been uploaded.
Run `python scripts/stage_peerj_archives.py` to rebuild the deterministic data,
checkpoint, result, and Figure 5 ZIP files. The source archive is made from the
final Git candidate separately so it represents an exact committed tree.

ZIP payloads and clean-clone verification workspaces are intentionally ignored
by Git. `SHA256SUMS.txt` records every staged archive and critical standalone
artifact.
