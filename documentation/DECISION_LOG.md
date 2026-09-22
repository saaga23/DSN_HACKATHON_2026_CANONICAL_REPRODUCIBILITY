# Decision Log

Major research decisions only.

| Phase | Decision | Rationale |
|-------|----------|-----------|
| PHASE5 | Baseline established | CatBoost canonical pipeline chosen as reference |
| PHASE6 | Tail correction rejected | Tail adjustment did not improve OOF RMSE meaningfully |
| PHASE11 | E3 promoted internally, rejected publicly | E3 public RMSE 1084.00044 worse than baseline 1073.18868 |
| PHASE12 | Generalization failure investigated | E3 failed on test set - overfitting to validation |
| PHASE15 | Product-Store Holdout 2570 RMSE invalidated | RMSE calculation bug; result is INVALID |
| PHASE17 | Model identity error discovered | P17-S2-02/03 were CatBoost predictions mislabeled as LGBM/XGBoost |
| PHASE17A | Phase 17 claims corrected | "1 product=1 store" false, "all models identical" false, only 3 seeds |
| PHASE17B | Research reopened | All valid branches re-tested; outcome C (no useful signal) |
| PHASE18 | Ensemble branch rejected | All 48 blends worse than CatBoost; no complementary errors |
| Final | Submission frozen | FINAL_SUBMISSION.csv SHA256 verified, no new submission authorized |

---