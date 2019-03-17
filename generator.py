import os
import sys

from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #delete this if you want to use GPU

if __name__=='__main__':
    if len(sys.argv)==3:
        model = load_model(sys.argv[1])
        noise = np.random.uniform(-1, 1, size=(1, 1, 1, 100))
        img = model.predict(noise)[0]
        plt.imshow(img)
        if not os.path.exists('./generated_images/'):
            os.mkdir('generated_images')
        plt.savefig('./generated_images/'+sys.argv[2])
    else:
        print('Wrong number of arguments!')