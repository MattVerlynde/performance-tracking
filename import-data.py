import numpy as np
import os
# from matplotlib import pyplot as plt
# from plotly.io import write_html
# import plotly.express as px

name_data = np.sort(os.listdir("../../../Data/Selection"))
print(name_data)
# for f in name_data[12:]:
#     af = np.load('/home/verlyndem/Data/Selection/'+f)
#     print(np.unique(af))
#     if len(af.shape)==3:
#         # pass
#         fig, ax = plt.subplots(nrows=2, ncols=9)
    
#         for i in range(2):
#             for j in range(9):
#                 if 9*i+j < af.shape[2]:  
#                     im = ax[i, j].imshow(af[:,:,9*i+j])
#                     ax[i, j].axis('off')
#                 else:
#                     ax[i, j].axis('off')
#         fig.colorbar(im, ax=ax.ravel().tolist())
#         fig.suptitle(f[:-4])


#     else:
#         figure = px.imshow(af, aspect='auto',title=f[:-4])
#         figure.write_html(f[:-4]+'.html',include_mathjax='cdn')
#         im = plt.imshow(af)
#         plt.title(f[:-4])
#         plt.colorbar(im)
#     plt.savefig(f[:-4]+'.png', bbox_inches='tight', format="png")
#     plt.show()

f = 'Scene_4.npy'
image = np.load('/home/verlyndem/Data/Selection/'+f)
print(image.shape)

nf = '500x1000x68'
x = 500
y = 1000
z = 68
DIR = '/home/verlyndem/Data/Selection/'+nf
if not os.path.exists(DIR):
    os.mkdir(DIR)
for t in range(z):
    print(image.shape)
    new_image = image[:x,:y,:,t]
    np.save(os.path.join(DIR,nf+'_'+str(t)+'.npy'), new_image)
    print(new_image.shape)