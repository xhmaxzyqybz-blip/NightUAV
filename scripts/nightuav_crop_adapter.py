"""Strict-zero-only, same-image crop resampling. No brightness threshold."""
import numpy as np

class BlackCropRetryError(RuntimeError):
    def __init__(self, record):
        self.record=record
        super().__init__(f"Strict-zero crop retry limit exhausted: {record['source']} ({record['resample_count']} retries)")

def select_crop(rgb_chw, source, rng, initial_xy=None, size=256, max_retries=50):
    """Initial draw + at most 50 replacement draws; accept replacement #50 if valid.

    RGB must be a CPU CHW numpy view; no mutation. rng is caller-owned.
    No image substitution, skipping, epsilon, or minimum mean condition.
    """
    if rgb_chw.ndim!=3 or rgb_chw.shape[0]!=3:raise ValueError('Expected RGB CHW')
    h,w=rgb_chw.shape[1:]
    if min(h,w)<size:raise ValueError(f'Image smaller than crop: {source}')
    def draw():return rng.randint(0,w-size),rng.randint(0,h-size)
    x,y=initial_xy if initial_xy is not None else draw()
    record={'source':source,'initial_xywh':[x,y,size,size],'invalid_crops':[],'resample_count':0,'accepted_xywh':None}
    for attempt in range(max_retries+1):
        if not(0<=x<=w-size and 0<=y<=h-size):raise ValueError('Out-of-bounds crop')
        if np.any(rgb_chw[:,y:y+size,x:x+size]!=0):
            record['resample_count']=attempt;record['accepted_xywh']=[x,y,size,size]
            return x,y,record
        record['invalid_crops'].append({'attempt':attempt,'xywh':[x,y,size,size]})
        record['resample_count']=attempt
        if attempt==max_retries:raise BlackCropRetryError(record)
        x,y=draw()