import json
from pathlib import Path


# Clean data from 'data-raw.json' to 'data.json'

data_path = Path(__file__).parent.parent

with open(data_path / 'data-raw.json') as f:
    data = json.load(f)

def clean_quaternion(p):
    del p['q_x']
    del p['q_y']
    del p['q_z']
    del p['q_w']


for e in data:
    del e['_id']
    del e['save']
    if 'collection' in e:
        del e['collection']
    
    for a in e['actions']:
        del a['save']
        del a['index']
        del a['step']
        del a['estimated_reward_std']
        if 'clamping_distance' in a:
            del a['clamping_distance']
        clean_quaternion(a['pose'])
        if 'final_pose' in a:
            clean_quaternion(a['final_pose'])
        clean_quaternion(a['images']['rc-v']['pose'])
        clean_quaternion(a['images']['rd-v']['pose'])
        if 'rd-after' in a['images']:
            del a['images']['rd-after']

        if 'rc-after' in a['images']:
            del a['images']['rc-after']

print(data[0])

with open(data_path / 'data.json', 'w') as outfile:
    json.dump(data, outfile)

print('Cleaned and written file.')
