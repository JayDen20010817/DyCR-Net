import pickle
import torch
import os
import numpy as np

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

class loader_audio:
    def __init__(self):
        self.name="audio"
        self.require=[]

    def prepare(self,input,opt):
        self.id ={
            "train":load_file("DyCR-Net-main/input/prepared_clean/" + "a_train_id"),
            "test":load_file("DyCR-Net-main/input/prepared_clean/" + "a_test_id"),
            "valid":load_file("DyCR-Net-main/input/prepared_clean/"+ "a_valid_id")
        }
        self.transform_audio_path = "DyCR-Net-main/audio_tensor"

    def get(self,result,mode,index):
        audio_path=os.path.join(
                self.transform_audio_path,
                "{}.npy".format(self.id[mode][index])
            )
        audio = torch.from_numpy(np.load(audio_path))
        result["audio"]=audio
    

    def getlength(self,mode):
        return len(self.id[mode])