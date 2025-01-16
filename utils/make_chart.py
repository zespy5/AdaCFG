import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
def sort_keya(ele1):
    string1 = ele1.stem.split('-')[0]
    return int(string1)

root = Path('scheduler_rescale_generate_results')
datas = [*root.glob('*/*/*')]
for imgs in tqdm(datas):
    gs = sorted([*imgs.glob('*.pt')], key=sort_keya)
    fig, ax = plt.subplots()
    for guid in gs:
        guidance = guid.stem.split('-')[0]
        scales = torch.load(guid, weights_only=False)[0].numpy()
        x = np.arange(1,len(scales)+1)
        ax.plot(x,scales,label=f'guidance {guidance}')
        ax.legend()
        ax.set_title('epsilon scale')
    chart_file = imgs/'chart.png'
    plt.tight_layout()
    plt.savefig(chart_file)  
    plt.close() 
