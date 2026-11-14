# PILOT configuration

Moved unchanged from the root `README.md` on 2026-11-13. The values are
defined in `src/auroralm/config.py`.

| Setting | Value |
| --- | ---: |
| Parameters | 9,994,960 |
| Vocabulary | 8,192 |
| Context length | 512 |
| Hidden size | 256 |
| Transformer blocks | 6 |
| Attention heads | 8 |
| Key/value heads | 2 |
| Feed-forward size | 688 |
| Source state dimensions | 171 |
| Regulator dimensions | 11 |
| Learned state latent | 16 |

`config.py` also contains an untrained 8-expert `MOE_PILOT` configuration and
the `SMALL` and `FULL` planning configurations. They are compute-ladder targets,
not evidence that those models have been trained.
