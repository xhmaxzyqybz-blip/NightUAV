# U3D train v2.1 strict-zero 256x256 crop audit

No training, model loading, loss changes or original-image writes. All 770 source hashes match the existing split manifest.

## Method and interpretation

Pillow RGB decoding matches the training loader. A black pixel has all three decoded uint8 RGB channels exactly zero. An integral image of the any-channel-nonzero mask tests EVERY legal 256x256 integer crop with stride 1, including bottom/right borders. No sampling is used to decide whether a black patch exists. Small brute-force tests independently verify integral-window indexing.

Random simulation: three independent Python Random streams (1149,2026,3407), 1000 draws per image per seed, manifest order, uniform inclusive x then y. These are risk simulations, not a replay of a particular training epoch. The exact probability counts all black-valid top-left positions / all legal positions; the dataset probability averages images equally.

Components are 8-connected clusters of VALID CROP TOP-LEFT coordinates. They identify separate feasible location clusters, not proven separate physical scenes. Bounding rectangles may contain holes and must not be interpreted as entirely black rectangles. Overlapping black crops are counted individually for sampling probability. Nonzero-pixel fraction describes available nonblack pixels; it does not certify semantic usefulness, exposure quality, or a fully nonblack 256x256 crop.

## Results

- Images containing at least one strict-zero patch: 12/770.
- Images hit by random simulation (union of seeds): 11.
- Sampled black crops: 999/2310000 (0.04324675%).
- Exact equal-image uniform-crop risk: 0.04185635%.
- Runtime: 216.74 seconds.

|Source|Black pixels %|Nonzero pixels %|Black-valid positions|Exact crop risk %|Location clusters|Seed1149/2026/3407 hits (each /1000)|
|---|---:|---:|---:|---:|---:|---|
|0015.jpg|14.7401|85.2599|119708|1.752827|1|24/20/17|
|0069.jpg|6.6485|93.3515|5772|0.084517|2|1/1/1|
|0156.jpg|4.6376|95.3624|11461|0.167818|1|1/1/3|
|0187.jpg|43.1708|56.8292|50450|0.738715|4|7/9/12|
|0323.jpg|63.2768|36.7232|727832|10.657295|7|108/105/107|
|0455.jpg|45.4996|54.5004|500495|7.328509|6|80/81/64|
|0488.jpg|13.7009|86.2991|56234|0.823408|2|10/8/5|
|0610.jpg|10.2994|89.7006|17179|0.251544|3|4/3/3|
|0649.jpg|3.0175|96.9825|135|0.001977|1|0/0/0|
|0651.jpg|43.1442|56.8558|657226|9.623446|3|95/97/108|
|0760.jpg|17.4245|82.5755|48481|0.709884|2|7/5/5|
|0798.jpg|5.3986|94.6014|6109|0.089451|1|2/1/4|

## Location details

Coordinates use a zero-based top-left origin; boxes are x,y,width,height. Full per-image records are in experiments/exp005a_black_crop_audit/per_image.json.

### 0015.jpg

- Valid-top-left cluster 1: [2864, 1640, 721, 265]; representative exact black crop [3396, 1640, 256, 256]; 119708 positions.
### 0069.jpg

- Valid-top-left cluster 1: [64, 1824, 65, 81]; representative exact black crop [64, 1824, 256, 256]; 5265 positions.
- Valid-top-left cluster 2: [394, 1876, 119, 29]; representative exact black crop [394, 1876, 256, 256]; 507 positions.
### 0156.jpg

- Valid-top-left cluster 1: [3283, 1755, 302, 150]; representative exact black crop [3552, 1755, 256, 256]; 11461 positions.
### 0187.jpg

- Valid-top-left cluster 1: [3504, 0, 81, 337]; representative exact black crop [3552, 0, 256, 256]; 14256 positions.
- Valid-top-left cluster 2: [3277, 1536, 36, 49]; representative exact black crop [3288, 1536, 256, 256]; 1368 positions.
- Valid-top-left cluster 3: [2987, 1684, 262, 186]; representative exact black crop [3072, 1684, 256, 256]; 34476 positions.
- Valid-top-left cluster 4: [3334, 1856, 24, 15]; representative exact black crop [3334, 1856, 256, 256]; 350 positions.
### 0323.jpg

- Valid-top-left cluster 1: [3144, 552, 441, 753]; representative exact black crop [3153, 552, 256, 256]; 161739 positions.
- Valid-top-left cluster 2: [2032, 1344, 85, 17]; representative exact black crop [2032, 1344, 256, 256]; 1445 positions.
- Valid-top-left cluster 3: [368, 1440, 65, 33]; representative exact black crop [368, 1440, 256, 256]; 2145 positions.
- Valid-top-left cluster 4: [931, 1456, 94, 33]; representative exact black crop [931, 1456, 256, 256]; 3102 positions.
- Valid-top-left cluster 5: [1048, 1456, 2441, 449]; representative exact black crop [2400, 1456, 256, 256]; 420139 positions.
- Valid-top-left cluster 6: [704, 1491, 161, 93]; representative exact black crop [704, 1491, 256, 256]; 11645 positions.
- Valid-top-left cluster 7: [0, 1536, 785, 369]; representative exact black crop [128, 1536, 256, 256]; 127617 positions.
### 0455.jpg

- Valid-top-left cluster 1: [3104, 576, 81, 193]; representative exact black crop [3136, 576, 256, 256]; 10641 positions.
- Valid-top-left cluster 2: [3496, 736, 25, 33]; representative exact black crop [3520, 736, 256, 256]; 57 positions.
- Valid-top-left cluster 3: [3536, 784, 49, 41]; representative exact black crop [3536, 784, 256, 256]; 2009 positions.
- Valid-top-left cluster 4: [1888, 1064, 1697, 841]; representative exact black crop [2400, 1064, 256, 256]; 487060 positions.
- Valid-top-left cluster 5: [3456, 1088, 1, 15]; representative exact black crop [3456, 1088, 256, 256]; 15 positions.
- Valid-top-left cluster 6: [2784, 1128, 129, 9]; representative exact black crop [2800, 1128, 256, 256]; 713 positions.
### 0488.jpg

- Valid-top-left cluster 1: [2768, 8, 49, 25]; representative exact black crop [2768, 8, 256, 256]; 1225 positions.
- Valid-top-left cluster 2: [3056, 1696, 529, 209]; representative exact black crop [3472, 1696, 256, 256]; 55009 positions.
### 0610.jpg

- Valid-top-left cluster 1: [2784, 0, 97, 97]; representative exact black crop [2784, 0, 256, 256]; 1025 positions.
- Valid-top-left cluster 2: [3168, 64, 337, 159]; representative exact black crop [3492, 64, 256, 256]; 15311 positions.
- Valid-top-left cluster 3: [3520, 1874, 49, 31]; representative exact black crop [3568, 1874, 256, 256]; 843 positions.
### 0649.jpg

- Valid-top-left cluster 1: [3468, 1887, 11, 18]; representative exact black crop [3469, 1887, 256, 256]; 135 positions.
### 0651.jpg

- Valid-top-left cluster 1: [1312, 712, 2273, 1193]; representative exact black crop [3197, 712, 256, 256]; 650788 positions.
- Valid-top-left cluster 2: [3560, 876, 25, 245]; representative exact black crop [3584, 876, 256, 256]; 365 positions.
- Valid-top-left cluster 3: [448, 1824, 127, 81]; representative exact black crop [464, 1824, 256, 256]; 6073 positions.
### 0760.jpg

- Valid-top-left cluster 1: [3328, 1029, 49, 93]; representative exact black crop [3328, 1029, 256, 256]; 2103 positions.
- Valid-top-left cluster 2: [3392, 1234, 193, 463]; representative exact black crop [3392, 1234, 256, 256]; 46378 positions.
### 0798.jpg

- Valid-top-left cluster 1: [3360, 1832, 225, 61]; representative exact black crop [3496, 1832, 256, 256]; 6109 positions.

## 0069.jpg

Previously failed crop (68,1850,256,256) independently confirmed strict-zero. Exact random-crop probability: 0.084517%; simulation: 3/3000 (0.100000%). Nonzero pixels: 93.3515%.

## Limits and decision

This audit covers only the 770 training images, not dev, quarantine or official val/test. Strict zero differs from near-black; tiny positive means and other numerical risks are not audited. Random samples can miss rare black patches; the exhaustive check is authoritative for existence. Counts characterize current decoded RGB and the current 256x256 crop protocol. No crop policy is changed. Await human review before skipping, resampling, changing the objective, or restarting training.

## Supplemental continuous nonblack-region check

Only the 12 affected images were rechecked. A fully nonblack crop requires at least one nonzero RGB channel at EVERY pixel; exact integral occupancy sum = 65536. 11 of the 12 images have such fully nonblack crops. 0187.jpg has none under this strict every-pixel-nonzero definition, despite 56.8292% nonzero pixels overall; this does not imply that every crop is entirely black. This does not certify semantic usefulness, adequate brightness or enhancement quality.

|Source|Fully nonblack 256x256 positions|Fraction of legal crop positions %|Representative x,y,w,h|
|---|---:|---:|---|
|0015.jpg|3139355|45.9681|[2039, 1047, 256, 256]|
|0069.jpg|2439076|35.7142|[2341, 805, 256, 256]|
|0156.jpg|4414276|64.6361|[451, 708, 256, 256]|
|0187.jpg|0|0.0000|None|
|0323.jpg|534347|7.8242|[912, 321, 256, 256]|
|0455.jpg|1243928|18.2142|[825, 259, 256, 256]|
|0488.jpg|3517184|51.5004|[3084, 1048, 256, 256]|
|0610.jpg|4856625|71.1132|[2692, 1099, 256, 256]|
|0649.jpg|4250931|62.2443|[811, 1072, 256, 256]|
|0651.jpg|1616834|23.6745|[3357, 295, 256, 256]|
|0760.jpg|215566|3.1564|[559, 221, 256, 256]|
|0798.jpg|3381232|49.5098|[339, 525, 256, 256]|
