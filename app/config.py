import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_ROOT = os.path.join(BASE_DIR, 'data/')
SKETCH_VGG = os.path.join(DATA_ROOT, 'model/sketch_vgg16.pth')
PHOTO_FEATURE = os.path.join(DATA_ROOT, 'feature/photo-vgg16.pkl')
