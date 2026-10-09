# DAppSCAN balanced table — `dappscan_v3`

Made from `dappscan_v3_all.parquet` (fingerprint `2dac42463f25ac1f`); fingerprint `c827a997aef13543`.

## Balancing

Every labelled file is kept. In each fold, unlabelled files are randomly sampled down to 1 per labelled file (seed 42). This applies to train, validation **and test**, like the Slither pipeline's `balance_secure` (which kept 30%). Scores on this table are higher than on the natural test sets and must be read against the length-only floor measured on the same files.

| fold | labelled | unlabelled before | unlabelled kept | rows |
|---:|---:|---:|---:|---:|
| 0 | 175 | 1,750 | 175 | 350 |
| 1 | 196 | 1,521 | 196 | 392 |
| 2 | 180 | 1,603 | 180 | 360 |
| 3 | 174 | 2,079 | 174 | 348 |
| 4 | 193 | 1,621 | 193 | 386 |
| all | 918 | 8,574 | 918 | 1,836 |

## Classes

| idx | SWC | title | files | scored | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---:|---|---|---:|:---:|---:|---:|---:|---:|---:|
| 0 | SWC-135 | Code With No Effects | 203 | yes | 40 | 43 | 42 | 37 | 41 |
| 1 | SWC-101 | Integer Overflow and Underflow | 136 | yes | 27 | 28 | 28 | 25 | 28 |
| 2 | SWC-107 | Reentrancy | 113 | yes | 22 | 25 | 23 | 20 | 23 |
| 3 | SWC-102 | Outdated Compiler Version | 112 | yes | 22 | 24 | 23 | 20 | 23 |
| 4 | SWC-103 | Floating Pragma | 92 | yes | 18 | 20 | 19 | 16 | 19 |
| 5 | SWC-104 | Unchecked Call Return Value | 74 | yes | 15 | 16 | 15 | 13 | 15 |
| 6 | SWC-114 | Transaction Order Dependence | 74 | yes | 15 | 16 | 15 | 13 | 15 |
| 7 | SWC-128 | DoS With Block Gas Limit | 73 | yes | 15 | 16 | 14 | 13 | 15 |
| 8 | SWC-100 | Function Default Visibility | 42 | yes | 9 | 9 | 8 | 7 | 9 |
| 9 | SWC-131 | Presence of unused variables | 40 | yes | 8 | 8 | 8 | 8 | 8 |
| 10 | SWC-105 | Unprotected Ether Withdrawal | 37 | yes | 8 | 8 | 7 | 6 | 8 |
| 11 | SWC-116 | Block values as a proxy for time | 36 | yes | 8 | 8 | 7 | 6 | 7 |
| 12 | SWC-108 | State Variable Default Visibility | 31 | yes | 7 | 7 | 6 | 5 | 6 |
| 13 | SWC-113 | DoS with Failed Call | 24 | yes | 5 | 5 | 5 | 4 | 5 |
| 14 | SWC-119 | Shadowing State Variables | 21 | yes | 4 | 4 | 4 | 5 | 4 |
| 15 | SWC-120 | Weak Sources of Randomness from Chain Attributes | 13 | no | 4 | 3 | 2 | 2 | 2 |
| 16 | SWC-123 | Requirement Violation | 12 | no | 3 | 3 | 2 | 2 | 2 |
| 17 | SWC-129 | Typographical Error | 10 | no | 4 | 2 | 1 | 2 | 1 |
| 18 | SWC-126 | Insufficient Gas Griefing | 9 | no | 2 | 2 | 2 | 1 | 2 |
| 19 | SWC-112 | Delegatecall to Untrusted Callee | 8 | no | 2 | 1 | 2 | 1 | 2 |
| 20 | SWC-110 | Assert Violation | 7 | no | 3 | 1 | 1 | 1 | 1 |
| 21 | SWC-111 | Use of Deprecated Solidity Functions | 7 | no | 1 | 1 | 1 | 1 | 3 |
| 22 | SWC-115 | Authorization through tx.origin | 7 | no | 1 | 3 | 1 | 1 | 1 |
| 23 | SWC-134 | Message call with hardcoded gas amount | 7 | no | 1 | 1 | 1 | 1 | 3 |
| 24 | SWC-122 | Lack of Proper Signature Verification | 5 | no | 1 | 0 | 1 | 2 | 1 |
| 25 | SWC-124 | Write to Arbitrary Storage Location | 5 | no | 0 | 2 | 1 | 1 | 1 |
| 26 | SWC-125 | Incorrect Inheritance Order | 4 | no | 1 | 1 | 0 | 1 | 1 |
| 27 | SWC-106 | Unprotected SELFDESTRUCT Instruction | 3 | no | 1 | 0 | 1 | 0 | 1 |
| 28 | SWC-117 | Signature Malleability | 3 | no | 1 | 0 | 1 | 1 | 0 |
| 29 | SWC-118 | Incorrect Constructor Name | 3 | no | 0 | 2 | 0 | 0 | 1 |
| 30 | SWC-121 | Missing Protection against Signature Replay Attacks | 3 | no | 1 | 1 | 0 | 0 | 1 |
| 31 | SWC-133 | Hash Collisions With Multiple Variable Length Arguments | 3 | no | 1 | 0 | 0 | 1 | 1 |
| 32 | SWC-132 | Unexpected Ether balance | 2 | no | 0 | 0 | 1 | 1 | 0 |
| 33 | SWC-109 | Uninitialized Storage Pointer | 1 | no | 1 | 0 | 0 | 0 | 0 |

All 34 classes are trained. F1 is reported over the 15 classes with at least 20 files.

