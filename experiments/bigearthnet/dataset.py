from torch.utils.data import Dataset
from skimage import io
import os
import numpy as np
import json
import torch

a_mean = np.array([340.76769064, 429.9430203, 614.21682446,
                     590.23569706, 950.68368468, 1792.46290469,
                    2075.46795189,2218.94553375,2266.46036911, 
                    2246.0605464,1594.42694882,1009.32729131],dtype='float32').reshape(12,1,1)

a_std = np.array([554.81258967,572.41639287,582.87945694,
                675.88746967,729.89827633,1096.01480586,
                     1273.45393088,1365.45589904,1356.13789355,
                     1302.3292881,1079.19066363,818.86747235],dtype='float32').reshape(12,1,1)

BigEarthNet_43_labels = ["Continuous urban fabric", "Discontinuous urban fabric", "Industrial or commercial units", \
            "Road and rail networks and associated land", "Port areas", "Airports", "Mineral extraction sites", "Dump sites", \
            "Construction sites", "Green urban areas", "Sport and leisure facilities", "Non-irrigated arable land", "Permanently irrigated land", \
            "Rice fields", "Vineyards", "Fruit trees and berry plantations", "Olive groves", "Pastures", "Annual crops associated with permanent crops", \
            "Complex cultivation patterns", "Land principally occupied by agriculture, with significant areas of natural vegetation", \
            "Agro-forestry areas", "Broad-leaved forest", "Coniferous forest", "Mixed forest", "Natural grassland", \
            "Moors and heathland", "Sclerophyllous vegetation", "Transitional woodland/shrub", "Beaches, dunes, sands", \
            "Bare rock", "Sparsely vegetated areas", "Burnt areas", "Inland marshes", "Peatbogs", "Salt marshes", "Salines", \
            "Intertidal flats", "Water courses", "Water bodies", "Coastal lagoons", "Estuaries", "Sea and ocean"]


class BigEarthNet_Dataset(torch.utils.data.Dataset):
    def __init__(self,IMAGE_FOLDER,LABEL_FOLDER, cache=False, transforms=None):
        super(BigEarthNet_Dataset, self).__init__()
        self.Image_folder_dir = os.listdir(IMAGE_FOLDER)
        self.indices=list(range(len(self.Image_folder_dir)))
        self.IMAGE_FOLDER = IMAGE_FOLDER
    
    def __getitem__(self, Im_Id):

        Im_Files=self.IMAGE_FOLDER +'/'+self.Image_folder_dir[self.indices[Im_Id]]
        Sen_Im = np.asarray((io.imread(Im_Files)),dtype='float32')
        Sen_Im = Sen_Im.transpose(2,0,1)

        Sen_Im = np.true_divide(np.subtract(Sen_Im, a_mean), a_std)
        lab = np.zeros((43),dtype = 'float32')

        Json_path = self.Image_folder_dir[self.indices[Im_Id]]
        Json_File = self.IMAGE_FOLDER+'/' + Json_path[:-4] + '_labels_metadata.json'
        json_data = json.load(open(Json_File, 'r'))
        f_labels = json_data['labels']
 
        # TODO FIX THIS OVERLOAD OF LABEL CONVERSION

        for idx in f_labels:
            lab[BigEarthNet_43_labels.index(idx)]=1 
        #BigEartNet_19_labels_multiHot = cls2multiHot(f_labels)#, self.label_indices)
        #https://gitlab.tubit.tu-berlin.de/rsim/bigearthnet-19-models/blob/master/pytorch_utils.py
        # Return the torch.Tensor values
        return (torch.from_numpy(Sen_Im), #torplot ch.from_numpy(Sen_Im).todevice(device)
                torch.from_numpy(lab))
    def __len__(self):
        return len(self.Image_folder_dir)