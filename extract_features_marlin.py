from marlin_pytorch import Marlin
import torch.multiprocessing as mp

from tqdm import tqdm
import torch
import os
import sys
import pandas as pd

base_path = '/home/surbhi/ximi/'

def read_file(path):
    try:
        with open(path) as f:
            dat = [i.strip('\n') for i in f.readlines()]
    except:
        return []
    return dat

def log(path, content):
    with open(path, 'a') as f:
        f.write(content)
        f.write('\n')
        
def load_model(feature_type):
    
    model = Marlin.from_file(f"marlin_vit_{feature_type}_ytf", f"marlin_models/marlin_vit_{feature_type}_ytf.encoder.pt")
    return model

def load_labels():

    labels = pd.read_csv('/home/surbhi/ximi/final_labels.csv')
    labels = labels[labels['label']!='SNP(Subject Not Present)']
    return labels

marlin_feature_type = 'large'
def main(marlin_feature_type, rank):
    model = load_model(marlin_feature_type)
    model = model.cuda()
#     labels = load_labels()
    _todo_ = read_file(f'todo{rank}.txt')
    errors = []
    processed = read_file(f'{marlin_feature_type}_processed_{rank}.txt')
        
#     todo = list(set([f['chunk'] for f in labels.to_dict(orient='records')]) - set(processed))
    todo = set(['chunks/' + f for f in _todo_]) - set(processed)
    proc = os.listdir(f'marlin_features_{marlin_feature_type}/')
    todo = list(set(todo) - set(['chunks/' + f.strip('.pt') for f in proc]))
#     print (todo)
    for vname in tqdm(todo):
        try:
#             print (vname)
            
            path = os.path.join(base_path, vname)
            print (path)
            features = model.extract_video(path, crop_face=True)
            # saving pt file
            
            torch.save(features, f"marlin_features_{marlin_feature_type}/{vname.split('/')[-1]}.pt")
            # logging
            log(f'{marlin_feature_type}_processed_{rank}.txt', vname)

        except Exception as e:
            
#           logging errors
            print (e)
            log(f'{marlin_feature_type}_errors_{rank}.txt', vname)
            
if __name__ == '__main__':
    args = sys.argv
    main('large', 'ESC')
#     num_processes = 8
#     model = MyModel()
    # NOTE: this is required for the ``fork`` method to work
#     model.share_memory()
#     processes = []
#     for rank in range(num_processes):
#         p = mp.Process(target=main, args=(args[1], rank))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()
    
    
