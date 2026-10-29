# EXP005A-0 completion review

10 epochs / 7700 updates completed; no further training launched. Each epoch covers the exact 770 train files once. 11 dev evaluations, four recorded strict reloads with matching checkpoint hashes, 40 saved dev outputs and eight montage files verified. No automatic best selection.

## Timing qualification

Recorded training/evaluation/checkpoint wall time: 160351.628 seconds (44.542 hours). This includes very long stalls and is NOT uninterrupted active training time. An epoch7 backward interval spans 16652.166 seconds; epoch8 transfer spans 130864.949 seconds. Sleep/suspension or resource contention may explain such intervals, but the logs do not establish the cause. Do not interpret these intervals as pure GPU computation or transfer speed. Epoch9 also contains time not accounted for by the narrowly timed phases. Original metrics are preserved without exclusions or corrections.

## Epoch average training objectives

|Epoch|Total|RD|VS|Recorded train seconds|
|---|---:|---:|---:|---:|
|1|0.02591477|0.02567744|0.00023733|279.073|
|2|0.02225981|0.02211409|0.00014572|268.226|
|3|0.02433569|0.02425749|0.00007821|279.012|
|4|0.02654704|0.02647659|0.00007044|293.039|
|5|0.02191821|0.02188753|0.00003067|927.185|
|6|0.02596639|0.02592935|0.00003704|965.172|
|7|0.02522386|0.02519830|0.00002556|18115.472|
|8|0.02481236|0.02479027|0.00002208|132622.972|
|9|0.02997650|0.02995451|0.00002198|5687.933|
|10|0.02788645|0.02786874|0.00001771|888.167|

## Preclamp >1 channel percentage

|Image|0|1|2|3|4|5|6|7|8|9|10|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0049.jpg|14.7294|5.0020|5.6374|8.7532|4.6514|6.3390|5.7309|4.1657|2.2790|3.1093|4.4214|
|0114.jpg|2.2637|0.9557|1.0554|1.3952|0.9255|1.1439|1.0769|0.8855|0.5720|0.7330|0.9125|
|0166.jpg|2.4667|0.9009|1.0260|1.4127|0.8811|1.1328|1.0598|0.8435|0.5116|0.6717|0.8721|
|0179.jpg|15.6780|10.6161|10.7023|13.0276|9.9603|11.5354|11.1321|10.1500|6.1277|9.0145|10.2935|
|0216.jpg|4.4340|2.7231|2.8555|3.4260|2.6357|3.0447|2.9379|2.6956|1.8415|2.3766|2.7141|
|0343.jpg|1.4689|0.6245|0.6807|0.9259|0.6022|0.7484|0.7049|0.5883|0.3835|0.4932|0.6008|
|0758.jpg|17.9847|9.6162|9.8871|12.9638|8.9722|10.8606|10.2889|8.9855|5.5328|7.7269|9.1790|
|0792.jpg|16.4273|9.6893|9.7022|12.8603|8.7977|10.8874|10.3682|9.1089|5.2920|7.6582|9.2998|

## Mean encoded luminance

|Image|0|1|2|3|4|5|6|7|8|9|10|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0049.jpg|0.7305|0.4631|0.4616|0.5127|0.4450|0.4623|0.4626|0.4102|0.4548|0.4256|0.4458|
|0114.jpg|0.5737|0.2771|0.2693|0.2985|0.2607|0.2660|0.2607|0.2062|0.3034|0.2479|0.2587|
|0166.jpg|0.5587|0.2614|0.2524|0.2799|0.2449|0.2503|0.2426|0.1881|0.2901|0.2326|0.2426|
|0179.jpg|0.6722|0.4152|0.4098|0.4451|0.3980|0.4080|0.4056|0.3570|0.4204|0.3834|0.3978|
|0216.jpg|0.5787|0.2900|0.2816|0.3100|0.2718|0.2766|0.2719|0.2191|0.3143|0.2589|0.2702|
|0343.jpg|0.5494|0.2503|0.2407|0.2670|0.2332|0.2380|0.2303|0.1759|0.2806|0.2215|0.2311|
|0758.jpg|0.7319|0.4773|0.4751|0.5216|0.4596|0.4748|0.4747|0.4250|0.4697|0.4414|0.4601|
|0792.jpg|0.6724|0.4150|0.4092|0.4458|0.3975|0.4097|0.4059|0.3570|0.4186|0.3831|0.3982|

## Median encoded luminance

|Image|0|1|2|3|4|5|6|7|8|9|10|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0049.jpg|0.6996|0.4097|0.4111|0.4577|0.3937|0.4040|0.4082|0.3540|0.4119|0.3727|0.3912|
|0114.jpg|0.5349|0.2319|0.2220|0.2463|0.2151|0.2176|0.2110|0.1560|0.2656|0.2040|0.2128|
|0166.jpg|0.5119|0.2078|0.1959|0.2179|0.1901|0.1933|0.1828|0.1280|0.2449|0.1801|0.1877|
|0179.jpg|0.5830|0.2852|0.2784|0.3078|0.2679|0.2703|0.2704|0.2159|0.3100|0.2539|0.2660|
|0216.jpg|0.5481|0.2484|0.2386|0.2642|0.2299|0.2319|0.2277|0.1734|0.2789|0.2179|0.2281|
|0343.jpg|0.5199|0.2161|0.2050|0.2277|0.1986|0.2017|0.1926|0.1379|0.2523|0.1885|0.1967|
|0758.jpg|0.6960|0.4062|0.4072|0.4541|0.3906|0.3991|0.4042|0.3494|0.4088|0.3693|0.3873|
|0792.jpg|0.5833|0.2853|0.2785|0.3082|0.2685|0.2705|0.2707|0.2161|0.3099|0.2541|0.2662|

## P95 encoded luminance

|Image|0|1|2|3|4|5|6|7|8|9|10|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0049.jpg|1.0000|0.9122|0.9043|0.9960|0.8793|0.9540|0.9437|0.8999|0.8101|0.8598|0.9061|
|0114.jpg|0.8249|0.5419|0.5493|0.6169|0.5269|0.5532|0.5558|0.4999|0.5198|0.5025|0.5284|
|0166.jpg|0.8188|0.5327|0.5418|0.6065|0.5208|0.5464|0.5502|0.4937|0.5157|0.4984|0.5242|
|0179.jpg|1.0000|1.0000|1.0000|1.0000|1.0000|1.0000|1.0000|1.0000|0.9688|1.0000|1.0000|
|0216.jpg|0.8893|0.6102|0.6180|0.6992|0.5907|0.6283|0.6265|0.5719|0.5718|0.5643|0.5948|
|0343.jpg|0.7204|0.4281|0.4317|0.4806|0.4147|0.4274|0.4317|0.3772|0.4292|0.3940|0.4134|
|0758.jpg|1.0000|1.0000|1.0000|1.0000|1.0000|1.0000|1.0000|1.0000|0.9674|1.0000|1.0000|
|0792.jpg|1.0000|1.0000|1.0000|1.0000|1.0000|1.0000|1.0000|1.0000|0.9490|0.9966|1.0000|

## Interpretation limits

No NaN/Inf/OOM was reported, gradients and parameter updates passed all steps, and none of the 40 saved outputs is entirely black or white. These checks do not establish visual quality. The 0179 montage shows remaining bright-region saturation; visual degradation and usability require human review across all eight montages. Neither reduced saturation nor unsupervised dev objective is an image-quality score. No best epoch conclusion.

Montages: results/demos/di_retinex_exp005a0/montage/. Loader benchmark: num_workers=0, pin_memory=False, persistent_workers=False, no prefetch_factor; 2.973 images/s versus pinned 2.755 images/s in the second pass. No speedup relative to the tested default (1.0x); worker candidates were skipped for insufficient free RAM. Semantic equality applies to the tested fixed crops.
