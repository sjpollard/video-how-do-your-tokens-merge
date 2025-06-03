from .vivit import apply_patch as vivit
from .vivit import apply_duplicate_patch as duplicate_vivit
from .timesformer import apply_patch as timesformer
from .timesformer import apply_duplicate_patch as duplicate_timesformer
from .motionformer import apply_patch as motionformer
from .motionformer import apply_duplicate_patch as duplicate_motionformer
from .videomae import apply_patch as videomae
from .videomae import apply_duplicate_patch as duplicate_videomae

__all__ = ['vivit', 'duplicate_vivit', 'timesformer', 'duplicate_timesformer', 
           'motionformer', 'duplicate_motionformer', 'videomae', 'duplicate_videomae']
