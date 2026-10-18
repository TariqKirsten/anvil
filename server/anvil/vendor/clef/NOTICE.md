# Clef (Cloudflare), Apache-2.0

`joint_schema_model.py` is Cloudflare's Clef reference implementation, unmodified: the record encoding (`encode_record`,
`collate_records`), the joint schema head (`JointSchemaHead`), `ClefModel` and `systemone`. `LICENSE` is its licence,
also unmodified.

- Source: https://huggingface.co/Cloudflare/clef (identical in https://huggingface.co/Cloudflare/clef-flash),
  revision 6a3ac5fc030a3f33f7fa860a1271886b64a6606d
- sha256 `joint_schema_model.py`: 0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3
- sha256 `LICENSE`: bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a

Keep both files byte-identical, so models trained by anvil load wherever Clef does; `tests/test_clef.py` checks the
digests. How anvil trains the head (the loss, learning rates, checkpoints) is anvil's own code in `anvil/tasks.py` and
`anvil/train.py`.
