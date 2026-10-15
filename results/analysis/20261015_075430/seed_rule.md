# Seed rule (pre-registered, docs/PROTOCOL.md)

| metric | baseline+inject mean ± SD | FusionEdge mean ± SD | paired d (1234, 1235, 1236) | result |
|---|---|---|---|---|
| EN | 6.7212 ± 0.0005 | 6.7198 ± 0.0021 | +0.0004, +0.0032, +0.0006 | no clear difference |
| AG | 0.0237 ± 0.0000 | 0.0237 ± 0.0000 | -0.0001, -0.0000, -0.0000 | no clear difference |
| SF | 0.0642 ± 0.0000 | 0.0643 ± 0.0001 | -0.0002, -0.0001, -0.0001 | FusionEdge |
| MI | 6.2652 ± 0.0087 | 6.1346 ± 0.0488 | +0.0947, +0.1803, +0.1170 | baseline+inject |
| SSIM_ir | 0.4924 ± 0.0004 | 0.4818 ± 0.0023 | +0.0083, +0.0128, +0.0109 | baseline+inject |
| SSIM_vi | 0.9063 ± 0.0005 | 0.9177 ± 0.0038 | -0.0079, -0.0152, -0.0113 | FusionEdge |
| MS_SSIM_ir | 0.4857 ± 0.0004 | 0.4701 ± 0.0042 | +0.0111, +0.0192, +0.0165 | baseline+inject |
| MS_SSIM_vi | 0.9256 ± 0.0003 | 0.9350 ± 0.0031 | -0.0065, -0.0122, -0.0094 | FusionEdge |

FusionEdge clearly better on 3/8; baseline+inject clearly better on 3/8; baseline+inject matches or beats on 5/8.

**Verdict: FusionEdge backbone NOT supported**
