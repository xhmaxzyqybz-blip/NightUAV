"""Render measured EXP005A curves as standalone SVG; no ranking or training."""
import csv,html,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/exp005a_baseline_30ep'
def chart(path,title,series,ylabel):
 colors=['#1769aa','#d14900','#27944f','#8e44ad','#b82d4b','#007f86','#807100','#555555'];values=[v for xs,ys in series.values() for v in ys if v is not None];assert values
 lo=min(0,min(values));hi=max(values);span=hi-lo or 1;lo-=span*.04;hi+=span*.04
 def xy(x,y):return 80+x/15*760,410-(y-lo)/(hi-lo)*325
 out=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="480" viewBox="0 0 1100 480">','<rect width="1100" height="480" fill="white"/>',f'<text x="80" y="30" font-size="20">{html.escape(title)}</text>',f'<text x="80" y="55" font-size="13">{html.escape(ylabel)}; diagnostics only, no best selection</text>']
 for j in range(6):
  v=lo+(hi-lo)*j/5;_,y=xy(0,v);out.extend([f'<path d="M80 {y:.2f}H840" stroke="#ddd"/>',f'<text x="72" y="{y+4:.2f}" text-anchor="end" font-size="12">{v:.4g}</text>'])
 for x in range(0,16,5):
  xx,_=xy(x,lo);out.append(f'<text x="{xx:.2f}" y="432" text-anchor="middle" font-size="12">{x}</text>')
 for i,(label,(xs,ys)) in enumerate(series.items()):
  color=colors[i%len(colors)];segments=[];current=[]
  for x,y in zip(xs,ys):
   if y is None:
    if current:segments.append(current);current=[]
   else:current.append(xy(x,y))
  if current:segments.append(current)
  for points in segments:out.append('<polyline fill="none" stroke="'+color+'" stroke-width="2" points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+'"/>')
  out.extend([f'<path d="M870 {85+i*25}h25" stroke="{color}" stroke-width="3"/>',f'<text x="905" y="{89+i*25}" font-size="13">{html.escape(label)}</text>'])
 out.extend(['<text x="450" y="460" font-size="14">Epoch</text>','</svg>'])
 with path.open('x',encoding='utf-8') as f:f.write('\n'.join(out))
def main():
 t=json.loads((EXP/'metrics/train_history.json').read_text())['epochs'];d=json.loads((EXP/'metrics/dev_history.json').read_text());assert 0<len(t)<=15 and [x['epoch'] for x in t]==list(range(1,len(t)+1)) and [x['epoch'] for x in d]==list(range(len(t)+1))
 folder=EXP/'curves';folder.mkdir(exist_ok=False);chart(folder/'epoch_loss.svg','Epoch mean original training objectives',{label:([x['epoch'] for x in t],[x[key] for x in t]) for label,key in [('total','mean_total'),('RD','mean_RD'),('VS','mean_VS')]},'Loss')
 sources=[x['source'] for x in d[0]['images']];rows=[]
 for stage in d:
  for x in stage['images']:
   rows.append({'epoch':stage['epoch'],'source':x['source'],'dark_count':x['dark_region_gain']['pixel_count'],'dark_mean_gain':x['dark_region_gain']['mean'],'bright_count':x['bright_region_gain']['pixel_count'],'bright_mean_gain':x['bright_region_gain']['mean'],'gt1_ratio':x['raw_gt1_ratio'],'mean_luminance':x['mean_luminance'],'median_luminance':x['median_luminance'],'p95_luminance':x['p95_luminance']})
 for key,title in [('dark_mean_gain','Dark-region mean gain (fixed input Y < 0.2)'),('bright_mean_gain','Bright-region mean gain (fixed input Y > 0.7)'),('gt1_ratio','Preclamp >1 channel ratio'),('mean_luminance','Mean output encoded luminance')]:
  chart(folder/(key+'.svg'),title,{Path(p).stem:([x['epoch'] for x in rows if x['source']==p],[x[key] for x in rows if x['source']==p]) for p in sources},key)
 with (folder/'per_image_curves.csv').open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 page=f'# EXP005A measured curves\n\nCompleted epochs: {len(t)} of target 15. Curves contain only fully completed epochs; partial training is excluded from epoch means. Epoch0 is random initialization. Region masks are fixed input masks; empty-region values remain null. No quality score or best epoch is defined.\n\n'
 for p in sorted(folder.glob('*.svg')):page+=f'## {p.stem}\n\n![{p.stem}]({p.name})\n\n'
 with (folder/'README.md').open('x',encoding='utf-8') as f:f.write(page)
 print('CURVES COMPLETE',str(folder),flush=True)
if __name__=='__main__':main()
