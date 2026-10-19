# Haiku judgments of NLA head decodes

1380 judgments. Rubric is 0-3; forced-choice chance is 50%. Balanced accuracy averages the hit rates when the true passage is A and when it is B, so the judge's preference for A can't move it off 50% by itself.

## Main groups
| Group | n | Rubric, true passage | Rubric, wrong passage | Difference [95% CI] | Score ≥2 true / wrong | Forced choice acc. [95% CI] | Balanced acc. | Chose A | Acc. when not low confidence |
|---|---|---|---|---|---|---|---|---|---|
| residual (positive control) | 20 | 1.65 | 0.00 | 1.65 [1.40, 1.90] | 60% / 0% | 100% [84%, 100%] | 100% | 70% | 100% (n=20) |
| heads, raw | 200 | 0.04 | 0.01 | 0.03 [-0.01, 0.06] | 0% / 0% | 57% [51%, 64%] | 59% | 63% | 60% (n=112) |
| heads, mean+ | 200 | 0.07 | 0.01 | 0.05 [0.02, 0.10] | 1% / 0% | 57% [51%, 64%] | 56% | 72% | 54% (n=111) |
| random, raw (negative control) | 20 | 0.05 | 0.00 | 0.05 [0.00, 0.15] | 0% / 0% | 50% [30%, 70%] | 46% | 85% | 56% (n=9) |
| random, mean+ (negative control) | 20 | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0% / 0% | 45% [26%, 66%] | 40% | 75% | 33% (n=9) |

## Head splits
| Group | n | Rubric, true passage | Rubric, wrong passage | Difference [95% CI] | Score ≥2 true / wrong | Forced choice acc. [95% CI] | Balanced acc. | Chose A | Acc. when not low confidence |
|---|---|---|---|---|---|---|---|---|---|
| raw: layers 0-6 | 53 | 0.04 | 0.02 | 0.02 [-0.04, 0.08] | 0% / 0% | 53% [40%, 66%] | 54% | 60% | 53% (n=32) |
| raw: layers 7-13 | 61 | 0.08 | 0.02 | 0.07 [0.00, 0.13] | 0% / 0% | 72% [60%, 82%] | 74% | 56% | 75% (n=36) |
| raw: layers 14-20 | 46 | 0.02 | 0.00 | 0.02 [0.00, 0.07] | 0% / 0% | 52% [38%, 66%] | 55% | 74% | 58% (n=19) |
| raw: layers 21-27 | 40 | 0.00 | 0.03 | -0.03 [-0.07, 0.00] | 0% / 0% | 48% [33%, 63%] | 46% | 63% | 48% (n=25) |
| raw: attends mostly to first token | 74 | 0.03 | 0.03 | 0.00 [-0.05, 0.05] | 0% / 0% | 54% [43%, 65%] | 56% | 66% | 54% (n=37) |
| raw: attends elsewhere | 126 | 0.05 | 0.01 | 0.04 [0.01, 0.09] | 0% / 0% | 60% [51%, 68%] | 61% | 60% | 63% (n=75) |
| raw: AR match rank 1 of 10 | 92 | 0.05 | 0.01 | 0.04 [-0.02, 0.09] | 0% / 0% | 63% [53%, 72%] | 63% | 63% | 59% (n=51) |
| raw: AR match rank 2-10 | 108 | 0.03 | 0.02 | 0.01 [-0.03, 0.06] | 0% / 0% | 53% [43%, 62%] | 56% | 62% | 61% (n=61) |
| raw: head norm >= 3 | 57 | 0.07 | 0.02 | 0.05 [-0.02, 0.12] | 0% / 0% | 63% [50%, 74%] | 63% | 61% | 61% (n=33) |
| raw: head norm < 3 | 143 | 0.03 | 0.01 | 0.01 [-0.02, 0.05] | 0% / 0% | 55% [47%, 63%] | 57% | 63% | 59% (n=79) |
| mean+: layers 0-6 | 53 | 0.08 | 0.00 | 0.08 [0.02, 0.15] | 0% / 0% | 51% [38%, 64%] | 54% | 74% | 50% (n=28) |
| mean+: layers 7-13 | 61 | 0.08 | 0.02 | 0.07 [-0.02, 0.18] | 2% / 0% | 64% [51%, 75%] | 62% | 66% | 63% (n=35) |
| mean+: layers 14-20 | 46 | 0.02 | 0.02 | 0.00 [-0.07, 0.07] | 0% / 0% | 50% [36%, 64%] | 50% | 70% | 41% (n=27) |
| mean+: layers 21-27 | 40 | 0.07 | 0.03 | 0.05 [0.00, 0.13] | 0% / 0% | 65% [50%, 78%] | 57% | 83% | 62% (n=21) |
| mean+: attends mostly to first token | 74 | 0.05 | 0.01 | 0.04 [0.00, 0.09] | 0% / 0% | 58% [47%, 69%] | 57% | 66% | 58% (n=36) |
| mean+: attends elsewhere | 126 | 0.07 | 0.02 | 0.06 [0.01, 0.11] | 1% / 0% | 57% [48%, 65%] | 56% | 75% | 52% (n=75) |
| mean+: AR match rank 1 of 10 | 130 | 0.07 | 0.02 | 0.05 [0.00, 0.11] | 1% / 0% | 59% [51%, 67%] | 55% | 75% | 58% (n=73) |
| mean+: AR match rank 2-10 | 70 | 0.06 | 0.01 | 0.04 [0.00, 0.10] | 0% / 0% | 54% [43%, 65%] | 56% | 66% | 47% (n=38) |
| mean+: head norm >= 3 | 57 | 0.12 | 0.00 | 0.12 [0.04, 0.23] | 2% / 0% | 70% [57%, 80%] | 64% | 75% | 65% (n=34) |
| mean+: head norm < 3 | 143 | 0.04 | 0.02 | 0.02 [-0.02, 0.06] | 0% / 0% | 52% [44%, 60%] | 53% | 71% | 49% (n=77) |

## Highest-rated raw head decodes
- text 3 L4H1: score 1 (wrong passage 0), forced choice right; attends "\n" " roommate" " sink"; matches: "kitchen", "sink"
- text 5 L9H25: score 1 (wrong passage 0), forced choice right; attends " wildfire" "Residents" " blaze"; matches: "wildfire", "fire behavior and community safety"
- text 11 L9H26: score 1 (wrong passage 0), forced choice right; attends "Construction" " his" " cheaper"; matches: "18th-century British canal"
- text 12 L5H24: score 1 (wrong passage 0), forced choice right; attends "import" ".sort" "('"; matches: "coding platform / code formatting"
- text 12 L11H12: score 1 (wrong passage 0), forced choice right; attends "import" " (\n" "\n"; matches: "Python tool output"
- text 12 L18H3: score 1 (wrong passage 0), forced choice right; attends " ascending" ".sort" "',"; matches: "Python", "code examples"
- text 14 L13H16: score 1 (wrong passage 0), forced choice right; attends "lock" " that" " air"; matches: "airlock", "space suit hatch", "door"
- text 19 L12H15: score 1 (wrong passage 0), forced choice right; attends "Hi" "." "ups"; matches: "email format", "design proposal", "formal email to a colleague", "professional correspondence"
- text 1 L0H0: score 0 (wrong passage 0), forced choice right; attends "." " CRA" ","; matches: 
- text 1 L6H20: score 0 (wrong passage 0), forced choice wrong; attends "\n" " bank" " contributed"; matches: 
- text 1 L11H6: score 0 (wrong passage 0), forced choice wrong; attends "\n" " CRA" " The"; matches: 
- text 1 L14H9: score 0 (wrong passage 0), forced choice right; attends "\n" " CRA" " The"; matches: 
- text 1 L14H11: score 0 (wrong passage 0), forced choice wrong; attends " year" "SA" " contributed"; matches: 
- text 1 L18H13: score 0 (wrong passage 0), forced choice wrong; attends "\n" " CRA" "SA"; matches: 
- text 1 L20H4: score 0 (wrong passage 0), forced choice wrong; attends "\n" " CRA" " The"; matches: 
