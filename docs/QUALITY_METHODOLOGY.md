# Quality assessment methodology

Three questions are never mixed:

1. **Model detection performance** — only against independently reviewed *roof* labels (`--reviewed-labels`): object precision/recall/F1 (greedy one-to-one at IoU ≥ 0.5), mean matched IoU, area IoU/precision/recall.
2. **Cadastral correspondence** — roof candidates vs cadastral construction / block / UC geometry (a different concept).
3. **Geometry & data quality** — validity, emptiness, non-polygon types, overlapping pairs, CRS declared, area statistics.

The JSON report groups measures by ISO 19157-1:2023 elements (logical consistency, completeness, thematic accuracy, positional accuracy, temporal quality) as a *conceptual organisation*; no ISO certification or municipal conformity is claimed. Positional accuracy and temporal quality are reported as **not assessed**.

## Matching (many-to-many)
Edge = candidate/reference pair with intersection ≥ `min_coverage` of the smaller geometry. Connected components define groups:

| Group | Class |
|---|---|
| 1 candidate ↔ 1 reference, IoU ≥ `strong_iou` | strong correspondence |
| 1 ↔ 1, lower IoU | partial correspondence |
| group with >1 candidate or >1 reference | ambiguous (one-to-many / many-to-one / many-to-many) |
| candidate without edge | candidate without reference |
| reference without edge | reference without candidate |
| outside the processed valid area | not evaluated |

Per pair: IoU, intersection area, coverage of both sides, absolute and percentage area difference. Thresholds are demonstration parameters, not official tolerances. Every construction unit is **not** assumed to correspond to one roof.
