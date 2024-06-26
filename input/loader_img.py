import pickle
import torch
import os
import numpy as np

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

class loader_img:
    def __init__(self):
        self.name="img"
        self.require=[]

    def prepare(self,input,opt):
        self.id ={
            "train":load_file("DyCR-Net-main/input/prepared_clean/" + "v_train_id"),
            "test":load_file("DyCR-Net-main/input/prepared_clean/" + "v_test_id"),
            "valid":load_file("DyCR-Net-main/input/prepared_clean/"+ "v_valid_id")
        }
        self.transform_image_path = "DyCR-Net-main/image_tensor"

    def get(self,result,mode,index):
        img_path=os.path.join(
                self.transform_image_path,
                "{}.npy".format(self.id[mode][index])
            )
        img = torch.from_numpy(np.load(img_path))
        result["img"]=img
    

    def getlength(self,mode):
        return len(self.id[mode])