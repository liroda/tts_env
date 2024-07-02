#!/usr/bin/env python3

import numpy as np


for i in range(0,330):
        a = np.random.randint(min,max,i)
        part_input_ids = intersperse(a.tolist(),0)
        x_tst_np = np.array(part_input_ids,dtype=np.int64).reshape(1,-1)
        x_tst_lengths_np = np.array([len(part_input_ids)],dtype=np.int64)
        input_file = os.path.join(outdir,'x_{}'.format(i))
        input_lengths_file = os.path.join(outdir,'x_length_{}'.format(i))
        if not os.path.exists(input_file):
            f1 = open(input_file,'wb')
            f2 = open(input_lengths_file,'wb')
            f1.write(x_tst_np.tobytes())
            f2.write(x_tst_lengths_np.tobytes())
            f1.close()
            f2.close()
