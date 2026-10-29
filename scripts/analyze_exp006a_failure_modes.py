"""EXP006A: frozen E15 saved-image diagnostics; no model load or training."""
import hashlib, html, io, json, math, statistics, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/exp005b_di_retinex_u3d_baseline'
RESULTS = ROOT / 'results/demos/di_retinex_exp005b'
EXP = ROOT / 'experiments/exp006a_failure_analysis'
REGIONS = ('dark', 'middle', 'bright')
PRIORITY = ('0049', '0179', '0758', '0792')
COEFFICIENTS = np.array([.2126, .7152, .0722], dtype=np.float32)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)

def dist(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {'count': 0, 'mean': None, 'std': None, 'min': None, 'max': None, 'p01': None, 'p05': None, 'p10': None, 'p25': None, 'median': None, 'p75': None, 'p90': None, 'p95': None, 'p99': None}
    quantiles = np.quantile(values, [.01,.05,.1,.25,.5,.75,.9,.95,.99])
    result = {'count': int(values.size), 'mean': float(values.mean()), 'std': float(values.std()), 'min': float(values.min()), 'max': float(values.max())}
    result.update({key: float(value) for key,value in zip(['p01','p05','p10','p25','median','p75','p90','p95','p99'],quantiles)})
    return result

def transition(before, after, mask):
    incoming = before[mask]; outgoing = after[mask]; count = incoming.size
    new_count = int((~incoming & outgoing).sum()); old_count = int(incoming.sum())
    return {'denominator': count, 'input_count': old_count, 'input_ratio': old_count/count if count else None,
            'output_count': int(outgoing.sum()), 'output_ratio': float(outgoing.mean()) if count else None,
            'new_count': new_count, 'new_ratio_of_region': new_count/count if count else None,
            'new_ratio_of_input_unsaturated': new_count/(count-old_count) if count>old_count else None,
            'input_saturated_remaining_count': int((incoming & outgoing).sum()),
            'input_saturated_released_count': int((incoming & ~outgoing).sum())}

def colors(input_rgb, output_rgb, mask):
    before = input_rgb[mask].astype(np.float64); after = output_rgb[mask].astype(np.float64)
    sum_before = before.sum(1); sum_after = after.sum(1); eligible = (sum_before>0)&(sum_after>0)
    normalized_before = before[eligible]/sum_before[eligible,None]; normalized_after = after[eligible]/sum_after[eligible,None]
    delta = normalized_after-normalized_before
    if eligible.any():
        max_before = before[eligible].max(1); max_after = after[eligible].max(1)
        saturation_before = (max_before-before[eligible].min(1))/max_before
        saturation_after = (max_after-after[eligible].min(1))/max_after
    else:
        saturation_before = saturation_after = np.array([])
    return {'region_pixels': len(before), 'eligible_pixels': int(eligible.sum()), 'excluded_zero_sum_pixels': int((~eligible).sum()),
            'chromaticity_L1_change': dist(np.abs(delta).sum(1)),
            'signed_delta_r': dist(delta[:,0]), 'signed_delta_g': dist(delta[:,1]), 'signed_delta_b': dist(delta[:,2]),
            'HSV_S_input_mean': float(saturation_before.mean()) if eligible.any() else None,
            'HSV_S_output_mean': float(saturation_after.mean()) if eligible.any() else None,
            'HSV_S_delta': dist(saturation_after-saturation_before)}

def choose_rois(input_y):
    height,width = input_y.shape; size=128
    grid_x = sorted(set(range(0,width-size+1,64)) | {width-size})
    grid_y = sorted(set(range(0,height-size+1,64)) | {height-size})
    candidates=[]
    for top in grid_y:
        for left in grid_x:
            tile=input_y[top:top+size,left:left+size]
            candidates.append({'xywh':[left,top,size,size], 'input_mean_y':float(tile.mean()), 'input_dark_fraction':float((tile<.2).mean()), 'input_bright_fraction':float((tile>.7).mean())})
    selections=[]
    for category in ['highlight','dark','boundary']:
        eligible=[item for item in candidates if (item['input_bright_fraction']>=.1 if category=='highlight' else item['input_dark_fraction']>=.9 if category=='dark' else item['input_dark_fraction']>=.25 and item['input_bright_fraction']>=.05)]
        if not eligible:raise RuntimeError('No input-only eligible ROI: '+category)
        median=float(np.median([item['input_mean_y'] for item in eligible]))
        selected=min(eligible,key=lambda item:(abs(item['input_mean_y']-median), (item['xywh'][0]+64-width/2)**2+(item['xywh'][1]+64-height/2)**2, item['xywh'][1],item['xywh'][0]))
        selections.append({**selected,'category':category,'eligible_count':len(eligible),'eligible_median_input_y':median,'original_3840x2160_xywh':[value*3 for value in selected['xywh']]})
    return selections

def heatmap(gain):
    magnitude=np.minimum(np.abs(gain)/.25,1)[...,None]
    neutral=np.array([245,245,245]); positive=np.array([200,45,30]); negative=np.array([30,90,190])
    endpoint=np.where((gain>=0)[...,None],positive,negative)
    return np.rint(neutral+(endpoint-neutral)*magnitude).astype(np.uint8)

def png(path, image):
    with path.open('xb') as handle:image.save(handle,format='PNG')
    with Image.open(path) as reopened:reopened.verify()

def main():
    started=time.perf_counter()
    assert not EXP.exists(),'Refuse to overwrite existing EXP006A'
    frozen_checkpoint=ROOT/'checkpoints/di_retinex/exp005b/epoch015.pth'
    baseline_summary=read(BASE/'metrics/summary.json'); checkpoint_record=next(item for item in baseline_summary['checkpoint_checks'] if item['epoch']==15)
    checkpoint_hash=sha(frozen_checkpoint);assert checkpoint_hash==checkpoint_record['sha256']
    baseline_config=read(BASE/'config/run.json')
    paths=(ROOT/'configs/splits/u3d_dev_v2_1.txt').read_text().splitlines();assert len(paths)==len(set(paths))==8
    splits=read(ROOT/'configs/splits/u3d_train_dev_v2_1.json')
    fixed={item['source']:item for item in read(BASE/'config/fixed_dev_inputs.json')}
    float_e15={item['source']:item for item in read(BASE/'metrics/dev_epoch015.json')['images']}
    protected=[frozen_checkpoint,ROOT/'scripts/nightuav_crop_adapter.py']+list((ROOT/'configs/splits').glob('*v2_1*'))+list(BASE.rglob('*'))
    protected=[path for path in protected if path.is_file()]
    protected += [ROOT/'_model_audit/DI-retinex-low-light-enhancement'/name for name in ['mymodel.py','lowlight_train.py','dataloader.py']]
    before_hashes={path:sha(path) for path in protected}
    for source in paths:
        assert sha(ROOT/source)==splits['image_sha256'][source]
    (EXP/'config').mkdir(parents=True);(EXP/'metrics').mkdir();(EXP/'outputs').mkdir()
    config={'experiment':'EXP006A frozen E15 failure-mode diagnostics','baseline':'EXP005B','frozen_epoch':15,
            'checkpoint_path':frozen_checkpoint.relative_to(ROOT).as_posix(),'checkpoint_sha256':checkpoint_hash,
            'checkpoint_matches_baseline_record':True,'no_model_loaded':True,'no_inference_or_training':True,
            'script_sha256':sha(Path(__file__)),'analysis_resolution':[1280,720],'original_resolution':[3840,2160],
            'input_resize':'RGB Pillow LANCZOS, verified exact against archived input and fixed RGB hash',
            'Y_definition':'encoded RGB/255; Yprime = .2126 R + .7152 G + .0722 B, float32; no inverse sRGB transfer',
            'regions':{'dark':'Y_in < 0.2','middle':'0.2 <= Y_in <= 0.7','bright':'Y_in > 0.7'},
            'primary_high_saturation':'pixel with any uint8 RGB channel ==255','full_white':'all three RGB channels ==255',
            'low_endpoint':'any channel ==0; separately reported, not high saturation',
            'new_saturation':'input predicate false AND output predicate true; report /region pixels and /input unsaturated pixels',
            'quantization':'Saved clamped PNG only; 255 may include values rounded to255. Cannot distinguish actual float clipping from rounding per pixel.',
            'color_definition':'rgb chromaticity = RGB / sum(RGB); L1 delta and signed r/g/b deltas. Exclude sum==0 in either image, no epsilon. HSV S=(max-min)/max is color saturation, distinct from endpoint clipping.',
            'roi_rule':'Input only: 128x128 grid stride64 including right/bottom edges; highlight bright_fraction>=.10; dark dark_fraction>=.90; boundary dark_fraction>=.25 AND bright_fraction>=.05. Select closest meanY to eligible median, ties nearest image center then y,x. No output/gain used.',
            'roi_interpretation':'Illustrative input-stratified cases; not random or an exhaustive estimate of failure severity. No output-based selection.',
            'heatmap':'fixed diverging blue-negative / white-zero / red-positive; display range [-.25,+.25] shared by all crops, clipped display only; data unclipped',
            'gain_histogram_edges':np.linspace(-1,1,201).tolist(),'priority_images':list(PRIORITY),'protected_sha256':{path.relative_to(ROOT).as_posix():digest for path,digest in before_hashes.items()}}
    save_json(EXP/'config/run.json',config)
    region_records=[];color_records=[];sources=[];roi_records=[];overview_paths=[]
    for source in paths:
        name=Path(source).stem
        with Image.open(ROOT/source) as image:
            assert image.size==(3840,2160)
            input_u8=np.array(image.convert('RGB').resize((1280,720),Image.Resampling.LANCZOS))
        assert hashlib.sha256(input_u8.tobytes()).hexdigest()==fixed[source]['resized_rgb_sha256']
        with Image.open(RESULTS/'input'/(name+'.png')) as image:assert np.array_equal(input_u8,np.array(image))
        input_rgb=input_u8.astype(np.float32)/255;input_y=(input_rgb*COEFFICIENTS).sum(2)
        rois=choose_rois(input_y) if name in PRIORITY else []
        output_path=RESULTS/'epoch015'/(name+'.png')
        with Image.open(output_path) as image:
            assert image.mode=='RGB' and image.size==(1280,720)
            output_u8=np.array(image)
        output_rgb=output_u8.astype(np.float32)/255;output_y=(output_rgb*COEFFICIENTS).sum(2);gain=output_y-input_y
        mean_error=abs(float(output_y.mean())-float_e15[source]['mean_luminance']);assert mean_error<=.5/255+1e-6
        masks={'dark':input_y<.2,'middle':(input_y>=.2)&(input_y<=.7),'bright':input_y>.7}
        assert sum(int(mask.sum()) for mask in masks.values())==input_y.size
        assert int(masks['dark'].sum())==fixed[source]['dark_pixel_count'] and int(masks['bright'].sum())==fixed[source]['bright_pixel_count']
        modes={'high_any_channel':(np.any(input_u8==255,2),np.any(output_u8==255,2)),
               'high_all_channels_white':(np.all(input_u8==255,2),np.all(output_u8==255,2)),
               'low_any_channel':(np.any(input_u8==0,2),np.any(output_u8==0,2)),
               'low_all_channels_black':(np.all(input_u8==0,2),np.all(output_u8==0,2))}
        for region,mask in masks.items():
            count=int(mask.sum());clip={key:transition(before,after,mask) for key,(before,after) in modes.items()}
            channels_mask=np.repeat(mask[...,None],3,2)
            clip['high_channelwise']=transition(input_u8==255,output_u8==255,channels_mask)
            clip['low_channelwise']=transition(input_u8==0,output_u8==0,channels_mask)
            for item in clip.values():assert item['output_count']==item['new_count']+item['input_saturated_remaining_count']
            histogram=np.histogram(gain[mask],bins=np.linspace(-1,1,201))[0];assert int(histogram.sum())==count
            region_records.append({'source':source,'region':region,'pixel_count':count,'pixel_ratio':count/input_y.size,'input_luminance_mean':float(input_y[mask].mean()) if count else None,'output_luminance_mean':float(output_y[mask].mean()) if count else None,'gain':dist(gain[mask]),'gain_histogram_counts':histogram.tolist(),'endpoint_saturation':clip})
            color_records.append({'source':source,'region':region,**colors(input_rgb,output_rgb,mask)})
        sources.append({'source':source,'source_sha256':sha(ROOT/source),'saved_input_path':(RESULTS/'input'/(name+'.png')).relative_to(ROOT).as_posix(),'saved_input_sha256':sha(RESULTS/'input'/(name+'.png')),'e15_output_path':output_path.relative_to(ROOT).as_posix(),'e15_output_sha256':sha(output_path),'saved_png_mean_y_minus_baseline_float':float(output_y.mean())-float_e15[source]['mean_luminance'],'quantization_bound_check':True,'baseline_float_raw_gt1_channel_ratio':float_e15[source]['raw_gt1_ratio']})
        if rois:
            overview=Image.fromarray(input_u8);drawing=ImageDraw.Draw(overview);palette={'highlight':'yellow','dark':'cyan','boundary':'magenta'}
            for roi in rois:
                left,top,width,height=roi['xywh'];category=roi['category'];label=name+'_'+category
                drawing.rectangle((left,top,left+width-1,top+height-1),outline=palette[category],width=3);drawing.text((left+3,top+3),category,fill=palette[category],stroke_width=1,stroke_fill='black')
                tile_gain=gain[top:top+height,left:left+width]
                panel=Image.new('RGB',(768,366),'#20242a');pen=ImageDraw.Draw(panel)
                pen.text((8,7),f'{name} / {category} | xywh={roi["xywh"]} at 1280x720',fill='white')
                parts=[input_u8[top:top+height,left:left+width],output_u8[top:top+height,left:left+width],heatmap(tile_gain)]
                for column,(part,title) in enumerate(zip(parts,['Input','Frozen E15 PNG','Delta Y (output - input)'])):
                    pen.text((column*256+8,28),title,fill='white');panel.paste(Image.fromarray(part).resize((256,256),Image.Resampling.NEAREST),(column*256,48))
                bar=Image.fromarray(heatmap(np.tile(np.linspace(-.25,.25,220,dtype=np.float32),(12,1))));panel.paste(bar,(526,323))
                pen.text((520,342),'-0.25',fill='white');pen.text((633,342),'0',fill='white');pen.text((714,342),'+0.25',fill='white');pen.text((8,316),'Fixed scale; blue decrease / red increase',fill='white');pen.text((8,338),'2x nearest display; data at native analysis resolution',fill='white')
                filename=label+'.png';png(EXP/'outputs'/filename,panel)
                roi_records.append({'source':source,**roi,'panel_path':'outputs/'+filename,'heatmap_beyond_display_range_ratio':float((np.abs(tile_gain)>.25).mean())})
            filename=name+'_input_coordinates.png';png(EXP/'outputs'/filename,overview);overview_paths.append({'source':source,'path':'outputs/'+filename})
        print('ANALYZED',name,flush=True)
    pooled=[]
    for region in REGIONS:
        members=[item for item in region_records if item['region']==region];count=sum(item['pixel_count'] for item in members)
        pooled.append({'region':region,'pixel_count':count,'fraction_of_all_8_images':count/(8*1280*720),'input_mean':sum(item['input_luminance_mean']*item['pixel_count'] for item in members)/count,'output_mean':sum(item['output_luminance_mean']*item['pixel_count'] for item in members)/count,'mean_gain':sum(item['gain']['mean']*item['pixel_count'] for item in members)/count,'input_high_saturation_ratio':sum(item['endpoint_saturation']['high_any_channel']['input_count'] for item in members)/count,'output_high_saturation_ratio':sum(item['endpoint_saturation']['high_any_channel']['output_count'] for item in members)/count,'new_high_saturation_ratio':sum(item['endpoint_saturation']['high_any_channel']['new_count'] for item in members)/count})
    save_json(EXP/'config/sources.json',sources);save_json(EXP/'config/crop_coordinates.json',roi_records)
    save_json(EXP/'metrics/regions.json',region_records);save_json(EXP/'metrics/color_changes.json',color_records);save_json(EXP/'metrics/pooled_regions.json',pooled)
    document='<!doctype html><html lang="en"><meta charset="utf-8"><title>NightUAV EXP006A frozen E15 review</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px;background:#f5f6f8;color:#20242a}img{max-width:100%;height:auto}figure{margin:24px 0;padding:16px;background:white}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #bbb}code{word-break:break-all}</style><h1>EXP006A: frozen E15 review</h1><p>No quality ranking or hypothesis confirmation. Saved 8-bit outputs; endpoint saturation includes rounding.</p><p>Checkpoint SHA256: <code>'+checkpoint_hash+'</code></p><p>All crops chosen from input only, using fixed eligibility and median-brightness rule. Heatmaps share [-0.25,+0.25]; white=0, blue=decrease, red=increase. Coordinates refer to 1280x720; multiply by3 for original images.</p>'
    for item in overview_paths:
        name=Path(item['source']).stem;document+=f'<h2>{name}</h2><figure><img src="{html.escape(item["path"].split("/",1)[1])}" alt="Input-only crop locations for {name}"><figcaption>Input coordinates: yellow highlight, cyan dark, magenta boundary.</figcaption></figure>'
        for roi in roi_records:
            if roi['source']==item['source']:document+=f'<figure><img src="{html.escape(roi["panel_path"].split("/",1)[1])}" alt="{name} {roi["category"]} input output gain"><figcaption>{roi["category"]}: analysis xywh={roi["xywh"]}; original xywh={roi["original_3840x2160_xywh"]}; input-only eligible windows={roi["eligible_count"]}.</figcaption></figure>'
    document+='<p>Review regional metrics and original scenes before interpreting clipping, noise, color, or boundary artifacts. No GT quality claim.</p></html>'
    with (EXP/'outputs/index.html').open('x',encoding='utf-8') as handle:handle.write(document)
    notes='# Crop coordinates and selection\n\nAll positions chosen from input Y only, before reading the output PNG in the loop. Grid128, stride64, border positions included. Highlight: at least10% bright pixels; dark: at least90% dark pixels; boundary: at least25% dark and5% bright. Choose eligible median input meanY; ties favor image center then y,x. No output, gain, saturation or model failure is used for selection. These are illustrative cases, not a random sample of failures.\n\nBoxes are zero-based x,y,width,height at1280x720. Multiply each by3 for geometric coordinates in3840x2160 originals. Resized crops are not native4K evaluations; LANCZOS includes neighboring pixels. Display panels use nearest 2x scaling; all measurements use original1280x720 arrays. Shared heatmap limit +/-0.25 is visualization-only.\n\n|Image|Type|Analysis xywh|Original xywh|Eligible windows|\n|---|---|---|---|---:|\n'
    for roi in roi_records:notes+=f'|{Path(roi["source"]).name}|{roi["category"]}|{roi["xywh"]}|{roi["original_3840x2160_xywh"]}|{roi["eligible_count"]}|\n'
    with (EXP/'CROP_COORDINATES.md').open('x',encoding='utf-8') as handle:handle.write(notes)
    questions='# Human review checklist\n\nNo item is automatically confirmed.\n\n- [ ] Are input-only selected boxes representative enough, and are relevant objects visible? Record additional manual coordinates before examining output severity if more cases are needed.\n- [ ] Do added endpoint pixels remove visible highlight texture, or occur in already featureless/light-source areas? Distinguish input-existing and newly saturated pixels; no paired GT is available.\n- [ ] Does dark-region gain reveal structure, noise, or both? This experiment has no calibrated noise/detail ground truth.\n- [ ] Do boundary crops show halos or smooth transitions? The contact sheet alone does not prove an artifact mechanism.\n- [ ] Are normalized RGB changes visible as hue/chroma changes? Near-black ratios can be unstable; exact-black pixels are excluded and counts reported. Color difference is not a quality loss.\n- [ ] Would floating-point, prequantization outputs change endpoint-transition conclusions? Current PNG analysis cannot identify actual preclamp >1 pixels individually.\n- [ ] Are results consistent across all8 dev images rather than only the4 illustrated images? Regional JSON covers all8.\n- [ ] Review limitations of the8-image development split and 1280x720 evaluation before any generalization statement.\n\nH1 is not confirmed by this report. No theoretical defect or new loss is proposed.\n'
    with (EXP/'REVIEW_QUESTIONS.md').open('x',encoding='utf-8') as handle:handle.write(questions)
    report='# EXP006A: frozen E15 failure-mode diagnostics\n\nCompleted analysis of the8 fixed dev inputs and saved EXP005B E15 PNG outputs. No model load, inference, training, network/loss/weight change or split change. E15 is frozen by user instruction, not selected by this analysis.\n\nCheckpoint SHA256: `'+checkpoint_hash+'`. Matches EXP005B checkpoint validation record. Input JPEG hashes match split records; regenerated LANCZOS RGB arrays exactly match saved baseline input pixels and recorded RGB hashes. Output PNG hashes are frozen here. Their association with E15 relies on the recorded baseline export; no model re-inference was performed. Mean output Y agrees with baseline float records within the8-bit rounding bound.\n\n## Definitions\n\nAll pixel statistics are on1280x720 RGB arrays, Yprime=.2126R+.7152G+.0722B in encoded RGB/255. This is the baseline diagnostic definition, not physical linear luminance. Regions are determined exclusively by input: dark<.2, middle[.2,.7], bright>.7.\n\nPrimary saturation is ANY RGB channel equal to255 in saved PNG. Input-existing, output total, newly saturated (~input_sat & output_sat), retained and released saturation are separate counts. New saturation has two denominators: all region pixels and input-unsaturated region pixels. All-white, any-zero, all-black and channelwise endpoint statistics are also recorded separately. PNG255 can reflect rounding as well as clamping; do not label these counts as exact floating preclamp clipping.\n\nGain distributions include mean/std/min/max/p01/p05/p10/p25/median/p75/p90/p95/p99 and fixed[-1,1] histogram bins. Color metrics normalize RGB by R+G+B and report L1 and signed deltas; exact-zero sums excluded with counts, no epsilon. This removes common positive multiplicative RGB scaling, but is not a perceptual quality metric. HSV S reports color saturation, distinct from numeric endpoint clipping. Near-black color ratios remain sensitive to quantization.\n\n## Pooled region summary\n\nPixel-weighted across all8 dev images, not per-image arithmetic mean. High saturation means any channel255.\n\n|Region|Pixels|Input Y|Output Y|Gain|Input high %|Output high %|New high / region %|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for item in pooled:report+=f'|{item["region"]}|{item["pixel_count"]}|{item["input_mean"]:.6f}|{item["output_mean"]:.6f}|{item["mean_gain"]:.6f}|{100*item["input_high_saturation_ratio"]:.4f}|{100*item["output_high_saturation_ratio"]:.4f}|{100*item["new_high_saturation_ratio"]:.4f}|\n'
    report+='\n## Artifacts\n\n- config/run.json: all definitions, frozen checkpoint hash and protected source hashes.\n- config/sources.json: original, saved-input, E15-output hashes and float/PNG cross-check.\n- metrics/regions.json: all24 image-region records, distributions and transitions.\n- metrics/color_changes.json: all24 color-region records.\n- metrics/pooled_regions.json: pooled descriptive summary.\n- CROP_COORDINATES.md and config/crop_coordinates.json: input-only selection and coordinate mapping.\n- [Contact sheet](outputs/index.html):12 input/E15/gain comparisons and4 input overviews.\n- REVIEW_QUESTIONS.md: pending human questions, no automatic hypothesis confirmation.\n\nCrops were not selected for output severity. A changed endpoint fraction, gain or color statistic alone does not establish visual quality loss, causal mechanism, H1, or a theoretical defect of DI-Retinex. No innovation module or loss is proposed. Images remain under Git-ignored outputs/. No commit/push. Stop for human review.\n'
    with (EXP/'README.md').open('x',encoding='utf-8') as handle:handle.write(report)
    assert all(sha(path)==digest for path,digest in before_hashes.items())
    assert all(sha(ROOT/item['source'])==item['source_sha256'] and sha(ROOT/item['e15_output_path'])==item['e15_output_sha256'] for item in sources)
    assert len(region_records)==len(color_records)==24 and len(roi_records)==12
    save_json(EXP/'metrics/validation.json',{'status':'PASS','images':8,'image_regions':24,'crop_panels':12,'input_overviews':4,'checkpoint_sha256':checkpoint_hash,'protected_files_unchanged':True,'inputs_and_e15_outputs_unchanged':True,'regions_partition_each_image':True,'transition_count_identities_pass':True,'fixed_input_hashes_and_mask_counts_match':True,'saved_output_quantization_bound_pass':True,'seconds':time.perf_counter()-started})
    print('EXP006A ANALYSIS COMPLETE',json.dumps(pooled),flush=True)
if __name__=='__main__':main()