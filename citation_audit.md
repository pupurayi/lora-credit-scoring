# Citation Audit

Status as of citation verification pass. Three categories:
- **GREEN** — real paper, citation metadata correct or close enough that a minor fix is trivial. Keep.
- **AMBER** — real paper but citation metadata is wrong (author list, volume, venue, title). Keep paper, FIX metadata before submission.
- **RED** — paper not found, in predatory venue, or content claimed doesn't match. Replace or drop.

## GREEN — keep as-is or trivial fixes

| Key | Notes |
|-----|-------|
| `hu2021lora` | Canonical LoRA paper, arXiv 2106.09685. Correct. |
| `chen2016xgboost` | XGBoost SIGKDD 2016. Correct. |
| `lecun2015` | Deep learning, Nature 521. Correct. |
| `delong1988` | DeLong et al., Biometrics. Correct. |
| `mcnemar1947` | McNemar, Psychometrika. Correct. |
| `friedman2001` | Greedy function approximation. Correct. |
| `lundberg2017shap` | SHAP, NeurIPS 2017. Correct. |
| `stiglitz1981` | Credit rationing, AER. Correct. |
| `vaswani2017` | Attention is all you need. Correct. |
| `loshchilov2019adamw` | Decoupled weight decay. Correct. |
| `xu2019ctgan` | CTGAN, NeurIPS 2019. Correct. |
| `dua2017uci` | UCI repository. Correct. |
| `thomas2017` | Credit scoring textbook, SIAM. Correct. |
| `homecredit2018` | Kaggle competition. Correct. |
| `valipour2022dylora` | DyLoRA, EACL 2023 / arXiv 2210.07558. Correct. |
| `wang2023tabpeft` | arXiv 2309.06526 by Wang, Yu, Chen. Correct. |
| `stevens2020fairness` | IEEE SSCI 2020, Kiva paper. Correct. |
| `kazimoto2025` | Springer Human-Centric Intelligent Systems. Correct. |
| `ngwenya2024` | IEEE ICECCME 2024. Correct. |
| `finscope2022` | Survey exists. **BUT see RED section below for content issue.** |

## AMBER — real paper, metadata WRONG in current draft

| Key | What's wrong | Correction |
|-----|--------------|------------|
| `wang2025finlora` | Current authors "Wang Y, Zhang H, Li J, Chen X, Liu Y" are fabricated. | Real authors: **Dannong Wang, Jaisal Patel, Daochen Zha, Steve Y. Yang, Xiao-Yang Liu**. arXiv 2505.19819 confirmed. |
| `kim2024hydra` | Listed as Neural Networks Volume **179**, 106414. | Actual: Volume **178**, October 2024, article 106414. Authors correct. |
| `hlongwane2024` | Listed authors "Hlongwane NW, Bere A, Mangwende E" are fabricated. | Real authors: **Hlongwane R, Ramaboa KKKM, Mongwe W**. PLOS ONE 19(5) e0303566 / DOI 10.1371/journal.pone.0303566. |
| `bjorkegren2019` | Listed second author "Blumenstock J E" is wrong (confused with a different paper). | Real: **Björkegren, D., Grissen, D.** World Bank Econ Review 34(3), 618-634. |
| `speakman2018` | Listed venue "SIGKDD Workshop on Humanitarian Mapping". | Real venue: **ACM SIGCAS COMPASS 2018** (Conference on Computing and Sustainable Societies). |
| `chavan2023glora` | Author "Gupta D K" should be just "Gupta D"; "Xing E P" should be "Xing E". | Real authors: Chavan A, Liu Z, Gupta D, Xing E, Shen Z. |
| `hall2021fairlending` | Missing one author. | Real list: Hall P, Cox B, Dickerson S, Kannan AR, Kulkarni R, **Schmidt N**. |
| `alonso2022modelrisk` | Listed title "Machine learning in credit risk: measuring the dilemma between prediction and supervisory cost" is the title of a 2020 *working paper*, not the published 2022 article. | Real title: **"Measuring the model risk-adjusted performance of machine learning algorithms in credit default prediction"**. DOI 10.1186/s40854-022-00366-1 is correct. |
| `wang2022lift` | Listed authors "Wang T, Zhao J, Yatskar M, Chang KW, Ordonez V" do not match this paper at all (looks like authors of a different LIFT/bias paper). | Real authors: **Dinh T, Zeng Y, Zhang R, Lin Z, Gira M, Rajput S, Sohn J, Papailiopoulos D, Lee K**. NeurIPS 2022 / arXiv 2206.06565. |
| `olowe2021` | Pages listed as 1-12. | Actual pages: **17-22** in Int. J. Data Science & Technology 7(1). |
| `soundararajan2024` | Author listed as "Soundararajan B". | Real: **Balaji Soundararajan** (Independent Researcher). |

## RED — replace or drop

| Key | Reason | Action |
|-----|--------|--------|
| `oware2025` | EPRA International Journal of Multidisciplinary Research — paper not located in any indexer. EPRA itself is a known low-quality/predatory publisher. DOI `10.36713/epra23347` does not resolve cleanly. | **Drop.** Replace with one of the real fairness-in-credit papers (e.g. the MDPI 2025 SLR cited under `oware` candidates, or Hurlin & Pérignon 2024). |
| `tyagi2022xai` | Listed as "SSRN Preprint" with abstract id 4100036. Unverified by this audit pass. SSRN preprints by unaffiliated authors are weak citations. | **Verify or drop.** If retained, must verify the SSRN ID resolves to the cited content. |
| `nwaimo2024` | Real but published in *Computer Science & IT Research Journal* (fepbl.com) — borderline predatory, citation count negligible. | **Drop or replace** with a comparable peer-reviewed source. |
| `soundararajan2024` | Real but IJSREM is a known low-quality venue. | **Drop or replace** with the MIS Quarterly piece on AI-enabled credit scoring instead. |
| `shukla2024` | Not verified in this pass; ICCA 2024 unfamiliar venue. | **Verify in next pass.** |

## CRITICAL: the 48% claim is wrong

The abstract and intro state:

> "the Zimbabwe FinScope Consumer Survey 2022 reports that 48% of adults lack
> formal financial-service access"

This is **not what FinScope 2022 says**. The actual report finds that **financial inclusion in Zimbabwe rose to 88% in 2022** (up from 77% in 2014). Therefore only ~12% of adults are *fully* excluded.

The 48% number appears to have been hallucinated — possibly confused with an earlier survey or a different country.

**This is the central motivational statistic of the paper.** Every reference to it
needs to be replaced with a real, sourced statistic from FinScope 2022 such as:
- Financial inclusion: 88% (2022)
- Banked: ~30% (2022)
- Formal credit penetration: see Table from FinScope 2022 (need actual number)
- MSME credit barrier: real number to be looked up

We'll fix this in the rewrite. For now, **do not cite the 48% figure**.

## Summary scorecard

- 20 references checked
- 14 GREEN (keep, possibly with cosmetic touch-ups)
- 11 AMBER (real but metadata needs correction)
- 4-5 RED (drop or replace)
- 1 fabricated motivational statistic (the 48%)
- Remaining unverified: `tyagi2022xai`, `shukla2024`

Once we have real experimental results we'll also need to ADD citations for:
- Recent credit-scoring SLRs (MDPI 2025 SLR)
- Newer LoRA-for-tabular work (CTLoRA, Tab-PEFT successors)
- Bjorkegren/Blumenstock 2024 follow-up if it exists
- Real Zimbabwe-specific microfinance literature
