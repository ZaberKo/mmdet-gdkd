# %%
import pickle
from pathlib import Path
with Path("../work_dirs/teacher_gfl_r101_fpn_ms-2x_coco/20231014_013456/teacher_gfl_r101_fpn_ms-2x_coco.pkl").open('rb') as f:
    data=pickle.load(f)
# %%
data0=data[0]["pred_instances"]
data0['bboxes_logits'][0]

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
def plot_bbox_logits(bbox_logits_list, T=1):

    num_bbox=len(bbox_logits_list)
    fig=plt.figure(figsize=(4*4,3*num_bbox),dpi=300)
    cnt=1
    for i,bbox_logits in enumerate(bbox_logits_list):
        for j,bbox_logits_part in enumerate(bbox_logits):
            plt.subplot(num_bbox,4,cnt)
            cnt+=1
            p=F.softmax(bbox_logits_part/T)
            plt.bar(x=np.arange(p.shape[0]), height=p)
    fig.tight_layout()
    plt.show()


plot_bbox_logits(data0['bboxes_logits'][:5],T=4)
# %%
from analysis_tools.analyze_results import ResultVisualizer

ResultVisualizer()