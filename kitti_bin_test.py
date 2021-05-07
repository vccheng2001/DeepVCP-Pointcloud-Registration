import numpy as np

data = np.fromfile(str("000010.bin"), dtype=np.float32)
print(data.shape)