import json
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np

from utils.orthographic_image import OrthographicImage

def conv_ids_to_list(filepath):
    read1 = open(filepath,"r")
    lines = read1.readline()
    stringline = lines.splitlines()
    out = []
    temp = ""
    for i in range(0,len(lines)):
        if(stringline[0][i]!=','):
            temp += stringline[0][i]
        else:
            out.append(int(temp))
            temp = ""
            
    return out

class Loader:
    data_path = Path(__file__).parent.parent

    def __init__(self, data_path=None):
        self.data_path = Path(data_path) if data_path else self.data_path

        with open(self.data_path / 'datacc.json') as f:
            self.episodes = json.load(f)

    def __len__(self):
        return len(self.episodes)

    def get_episode(self, episode_index: int):

        return self.episodes[episode_index]

    def yield_episodes(self):
        for episode in self.episodes:
            yield episode

    @classmethod
    def get_image_path(cls, episode_id: str, action_id: int, camera: str):
        return cls.data_path / 'images' / episode_id / f'{action_id}-{camera}-v.png'

    def get_image(self, episode_index: int, action_id: int, camera: str, as_float=False):
       
        episode = self.episodes[episode_index]
        episode_id = episode['id']

        if camera[-2:] == 'cd':
            camera_key = camera[:-2]
            image_rc = self.get_image(episode_index, action_id, camera_key + 'c', as_float)
            image_rd = self.get_image(episode_index, action_id, camera_key + 'd', as_float)

            image_rc.mat = np.concatenate((image_rc.mat, np.expand_dims(image_rd.mat, axis=2)), axis=2)
            return image_rc

        image = cv2.imread(str(self.get_image_path(episode_id, action_id, camera)), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f'Image {episode_id} {action_id} {camera} not found.')

        # Make sure that image is either uint16 or float
        if not as_float and image.dtype == np.uint8:
            image = image.astype(np.uint16)
            image *= 255
        elif as_float:
            old_type = image.dtype
            image = image.astype(np.float32)
            image /= np.iinfo(old_type).max

        meta_data = episode['actions'][action_id]['images'][f'{camera}-v']

        return OrthographicImage(
            image,
            meta_data['info']['pixel_size'],
            meta_data['info']['min_depth'],
            meta_data['info']['max_depth'],
            camera,
            meta_data['pose'],
        )

    def get_action(self, episode_index: int, action_id, images: Union[str, List[str]] = None):
        episode = self.get_episode(episode_index)
        episode_id = episode['id']
        if not episode:
            raise Exception(f'Episode {episode_id} not found')
        action = episode['actions'][action_id]
        action['episode_id'] = episode_id
        if not images:
            return action

        if isinstance(images, str):
            images = [images]

        return (action, *[self.get_image(episode_index, action_id, camera) for camera in images])
    
    def remove_entry(self, episode_index:int, action_id,images: Union[str, List[str]] = None):
        #out = conv_ids_to_list(r"C:\Users\User\Documents\UNI\MEng Project\PYFILES\screwlogs.txt")
        #out_2 = conv_ids_to_list(r"C:\Users\User\Documents\UNI\MEng Project\PYFILES\screwlogs_0.txt")
        #out_3 = conv_ids_to_list(r"C:\Users\User\Documents\UNI\MEng Project\PYFILES\screwlogs_1.txt")
        with open(self.data_path/'datacc.json','r') as data_file:
            alldata = json.load(data_file)
            episode = self.get_episode(episode_index)
            episode_id = episode['id']
            action_id = self.get_action(episode_index, action_id)
            flag = False
            i=0
            for element in alldata:
                
                    #print(element)
                    try:
                        flag = (element['id']==episode_id)
                    except KeyError as ex:
                        continue
                    if (flag):
                        #del element['actions']
                        del element['id']
                        #element.pop('id')
                        del element['actions']
                        #element.pop('bin_episode_id',None)
                        try:
                            del element['bin_episode_id']
                        except: pass
                        #alldata.pop(i)
                        del alldata[i]
                        #print("Inside")
                        flag=False
                    #print("----------")
                    i+=1
        with open(self.data_path/'datacc.json', 'w') as data_file:
            alldata = json.dump(alldata,data_file)
        #print ("Episode ",episode_id," removed successfully")
        #print(self.get_episode(episode_index))
        
        