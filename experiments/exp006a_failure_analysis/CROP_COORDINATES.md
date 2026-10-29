# Crop coordinates and selection

All positions chosen from input Y only, before reading the output PNG in the loop. Grid128, stride64, border positions included. Highlight: at least10% bright pixels; dark: at least90% dark pixels; boundary: at least25% dark and5% bright. Choose eligible median input meanY; ties favor image center then y,x. No output, gain, saturation or model failure is used for selection. These are illustrative cases, not a random sample of failures.

Boxes are zero-based x,y,width,height at1280x720. Multiply each by3 for geometric coordinates in3840x2160 originals. Resized crops are not native4K evaluations; LANCZOS includes neighboring pixels. Display panels use nearest 2x scaling; all measurements use original1280x720 arrays. Shared heatmap limit +/-0.25 is visualization-only.

|Image|Type|Analysis xywh|Original xywh|Eligible windows|
|---|---|---|---|---:|
|0049.jpg|highlight|[192, 320, 128, 128]|[576, 960, 384, 384]|23|
|0049.jpg|dark|[1024, 256, 128, 128]|[3072, 768, 384, 384]|11|
|0049.jpg|boundary|[960, 448, 128, 128]|[2880, 1344, 384, 384]|30|
|0179.jpg|highlight|[1088, 512, 128, 128]|[3264, 1536, 384, 384]|77|
|0179.jpg|dark|[0, 320, 128, 128]|[0, 960, 384, 384]|61|
|0179.jpg|boundary|[256, 320, 128, 128]|[768, 960, 384, 384]|81|
|0758.jpg|highlight|[768, 256, 128, 128]|[2304, 768, 384, 384]|66|
|0758.jpg|dark|[0, 592, 128, 128]|[0, 1776, 384, 384]|21|
|0758.jpg|boundary|[640, 320, 128, 128]|[1920, 960, 384, 384]|47|
|0792.jpg|highlight|[448, 0, 128, 128]|[1344, 0, 384, 384]|59|
|0792.jpg|dark|[768, 0, 128, 128]|[2304, 0, 384, 384]|69|
|0792.jpg|boundary|[512, 0, 128, 128]|[1536, 0, 384, 384]|49|
