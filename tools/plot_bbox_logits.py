# %%
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


root=Path('../work_dirs/logits')
data_dict = {}

for p in root.iterdir():
    if p.is_file():
        data_dict[p.stem]=p

#%%
def plot_bbox_logits(bbox_logits_list, T=1, title=''):

    num_bbox=len(bbox_logits_list)
    fig=plt.figure(figsize=(4*4,3*num_bbox),dpi=300)
    plt.title(title)
    cnt=1
    for i,bbox_logits in enumerate(bbox_logits_list):
        for j,bbox_logits_part in enumerate(bbox_logits):
            plt.subplot(num_bbox,4,cnt)
            cnt+=1
            p=F.softmax(bbox_logits_part/T, dim=-1)
            plt.bar(x=np.arange(p.shape[0]), height=p)
    
    fig.tight_layout()
    fig.savefig(f'../work_dirs/logits/{title}_T{T}.png')
    plt.show()

#%%
with Path("../work_dirs/teacher_gfl_r101_fpn_ms-2x_coco/20231014_013456/teacher_gfl_r101_fpn_ms-2x_coco.pkl").open('rb') as f:
    data=pickle.load(f)
# %%


# %%


for name,path in data_dict.items():
    with path.open('rb') as f:
        data=pickle.load(f)

    data0=data[10]["pred_instances"]
    # data0['bboxes_logits'][0] 
    plot_bbox_logits(data0['bboxes_logits'][:5],T=2, title=name)



# %%
from analysis_tools.analyze_results import ResultVisualizer

ResultVisualizer()