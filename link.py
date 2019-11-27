import os
from sys import argv

dirPath = argv[1]

suffix = argv[2]

endDir = argv[3]

paths = [os.path.join(dirPath, p) for p in os.listdir(dirPath)]

for path in paths:
    prefix = os.path.basename(path).split('_')[0]
    os.system('ln -s {} {}/{}_{}'.format(path, endDir, prefix, suffix))
