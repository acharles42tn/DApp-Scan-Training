# DAppSCAN training table — `dappscan_v3_all`

Built 2026-10-09T03:16:50+00:00 from https://github.com/InPlusLab/DAppSCAN at `66a56619c447` (fingerprint `2dac42463f25ac1f`).

## What a row is

One unique Solidity file from `DAppSCAN-source/contracts` after cleaning. `labels` are the SWC classes below that auditors annotated in the file; `spans_json` gives the annotated line ranges in the cleaned text (used to label windows). An empty `labels` list means *no SWC weakness annotated* — not *secure*.

## Cleaning

DAppSCAN writes every label into the source as a `// SWC-…` comment. These are stripped, and every file's layout is canonicalised (tabs → 4 spaces, trailing whitespace and blank lines removed) so no trace of where a comment was remains; annotated line numbers are remapped.

| step | count |
|---|---:|
| Solidity files read | 21,457 |
| annotations (SWC instances) | 1,646 |
| `// SWC-…` label comments stripped from the source | 1,647 |
| unannotated empty files dropped | 2 |
| unannotated test/mock files dropped | 3,726 |
| unannotated interface-only files dropped | 4,147 |
| files in scope | 13,582 |
| unique after merging exact (normalised) duplicates | 9,492 |
| annotated files excluded (only rare classes) | 0 |
| **rows in the table** | 9,492 |
| rows with ≥1 class | 918 |
| rows with no annotated class | 8,574 |

## Classes

Kept: SWC classes annotated in ≥ 1 unique files. Folds are grouped by codebase (510 codebases; 168 merged through near-duplicate files, cap 300 rows → 342 groups) and stratified per class. Default split: test = fold 0, val = fold 1, train = the other three.

| idx | SWC | title | files | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | SWC-135 | Code With No Effects | 203 | 40 | 43 | 42 | 37 | 41 |
| 1 | SWC-101 | Integer Overflow and Underflow | 136 | 27 | 28 | 28 | 25 | 28 |
| 2 | SWC-107 | Reentrancy | 113 | 22 | 25 | 23 | 20 | 23 |
| 3 | SWC-102 | Outdated Compiler Version | 112 | 22 | 24 | 23 | 20 | 23 |
| 4 | SWC-103 | Floating Pragma | 92 | 18 | 20 | 19 | 16 | 19 |
| 5 | SWC-104 | Unchecked Call Return Value | 74 | 15 | 16 | 15 | 13 | 15 |
| 6 | SWC-114 | Transaction Order Dependence | 74 | 15 | 16 | 15 | 13 | 15 |
| 7 | SWC-128 | DoS With Block Gas Limit | 73 | 15 | 16 | 14 | 13 | 15 |
| 8 | SWC-100 | Function Default Visibility | 42 | 9 | 9 | 8 | 7 | 9 |
| 9 | SWC-131 | Presence of unused variables | 40 | 8 | 8 | 8 | 8 | 8 |
| 10 | SWC-105 | Unprotected Ether Withdrawal | 37 | 8 | 8 | 7 | 6 | 8 |
| 11 | SWC-116 | Block values as a proxy for time | 36 | 8 | 8 | 7 | 6 | 7 |
| 12 | SWC-108 | State Variable Default Visibility | 31 | 7 | 7 | 6 | 5 | 6 |
| 13 | SWC-113 | DoS with Failed Call | 24 | 5 | 5 | 5 | 4 | 5 |
| 14 | SWC-119 | Shadowing State Variables | 21 | 4 | 4 | 4 | 5 | 4 |
| 15 | SWC-120 ¹ | Weak Sources of Randomness from Chain Attributes | 13 | 4 | 3 | 2 | 2 | 2 |
| 16 | SWC-123 ¹ | Requirement Violation | 12 | 3 | 3 | 2 | 2 | 2 |
| 17 | SWC-129 ¹ | Typographical Error | 10 | 4 | 2 | 1 | 2 | 1 |
| 18 | SWC-126 ¹ | Insufficient Gas Griefing | 9 | 2 | 2 | 2 | 1 | 2 |
| 19 | SWC-112 ¹ | Delegatecall to Untrusted Callee | 8 | 2 | 1 | 2 | 1 | 2 |
| 20 | SWC-110 ¹ | Assert Violation | 7 | 3 | 1 | 1 | 1 | 1 |
| 21 | SWC-111 ¹ | Use of Deprecated Solidity Functions | 7 | 1 | 1 | 1 | 1 | 3 |
| 22 | SWC-115 ¹ | Authorization through tx.origin | 7 | 1 | 3 | 1 | 1 | 1 |
| 23 | SWC-134 ¹ | Message call with hardcoded gas amount | 7 | 1 | 1 | 1 | 1 | 3 |
| 24 | SWC-122 ¹ | Lack of Proper Signature Verification | 5 | 1 | 0 | 1 | 2 | 1 |
| 25 | SWC-124 ¹ | Write to Arbitrary Storage Location | 5 | 0 | 2 | 1 | 1 | 1 |
| 26 | SWC-125 ¹ | Incorrect Inheritance Order | 4 | 1 | 1 | 0 | 1 | 1 |
| 27 | SWC-106 ¹ | Unprotected SELFDESTRUCT Instruction | 3 | 1 | 0 | 1 | 0 | 1 |
| 28 | SWC-117 ¹ | Signature Malleability | 3 | 1 | 0 | 1 | 1 | 0 |
| 29 | SWC-118 ¹ | Incorrect Constructor Name | 3 | 0 | 2 | 0 | 0 | 1 |
| 30 | SWC-121 ¹ | Missing Protection against Signature Replay Attacks | 3 | 1 | 1 | 0 | 0 | 1 |
| 31 | SWC-133 ¹ | Hash Collisions With Multiple Variable Length Arguments | 3 | 1 | 0 | 0 | 1 | 1 |
| 32 | SWC-132 ¹ | Unexpected Ether balance | 2 | 0 | 0 | 1 | 1 | 0 |
| 33 | SWC-109 ¹ | Uninitialized Storage Pointer | 1 | 1 | 0 | 0 | 0 | 0 |
| | | **rows per fold** | | 1,925 | 1,717 | 1,783 | 2,253 | 1,814 |
| | | positive rows per fold | | 175 | 196 | 180 | 174 | 193 |

¹ Trained but not scored: fewer than 20 files, too few to evaluate. F1 is reported over the other 15 classes.

## Leakage check

Exact duplicates across folds: 0 (one row per normalised file). Near-duplicates (Jaccard ≥ 0.8 on 5-token shingles): 3,067 pairs, 956 crossing folds.

- test rows with a near-twin in train: **11.8%** (positive test rows: 2.3%)
- val rows with a near-twin in train: 3.3%

## Annotation line numbers

| parsed as | count |
|---|---:|
| beyond_eof | 2 |
| from_marker | 8 |
| marker | 8 |
| ranges | 1627 |
| whole | 11 |

