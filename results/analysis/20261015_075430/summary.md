# Holdout analysis (108 images, last.pt, holdout sha fbf10e110601)

Panels: IR | VI | baseline_simplecnn_inject | baseline_simplecnn_inject_s1235 | baseline_simplecnn_inject_s1236 | fusionedge_main | fusionedge_main_s1235 | fusionedge_main_s1236 | baseline_simplecnn_v6 | then gate, dY for each inject run in that order (JET: green = 0, red = brighter, ±0.3 full scale)

| metric | identity_vi | baseline_simplecnn_inject | baseline_simplecnn_inject_s1235 | baseline_simplecnn_inject_s1236 | fusionedge_main | fusionedge_main_s1235 | fusionedge_main_s1236 | baseline_simplecnn_v6 |
|---|---|---|---|---|---|---|---|---|
| EN | 6.6312 | 6.7213 | 6.7206 | 6.7216 | 6.7209 | 6.7174 | 6.7210 | 6.6885 |
| AG | 0.0217 | 0.0237 | 0.0237 | 0.0237 | 0.0238 | 0.0237 | 0.0237 | 0.0238 |
| SF | 0.0614 | 0.0642 | 0.0641 | 0.0642 | 0.0643 | 0.0642 | 0.0643 | 0.0642 |
| MI | 6.9863 | 6.2747 | 6.2634 | 6.2576 | 6.1800 | 6.0830 | 6.1406 | 3.6759 |
| SSIM_ir | 0.3930 | 0.4926 | 0.4927 | 0.4920 | 0.4843 | 0.4798 | 0.4811 | 0.5135 |
| SSIM_vi | 1.0000 | 0.9059 | 0.9062 | 0.9068 | 0.9138 | 0.9213 | 0.9181 | 0.9024 |
| MS_SSIM_ir | 0.3668 | 0.4859 | 0.4858 | 0.4853 | 0.4748 | 0.4667 | 0.4688 | 0.4996 |
| MS_SSIM_vi | 1.0000 | 0.9253 | 0.9257 | 0.9259 | 0.9318 | 0.9379 | 0.9353 | 0.9267 |
