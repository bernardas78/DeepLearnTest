import numpy as np

x = np.array([ [1,3,5 ], 
               [2,4,6]  ])

x1 = np.array([ [ [1e+0, 1e+4, 1e+8],
                  [1e+1, 1e+5, 1e+9]
                ], 
                [ [1e+2, 1e+6, 1e+10],
                  [1e+3, 1e+7, 1e+11]
                ]  
              ])

x2 = np.multiply(x,x1)
assert (x2[1][0][2] == 5e+10)
assert (x2[0][0][2] == 5e+8)

x = x.reshape(1,2,3)

x3 = np.multiply(x,x1)
assert (x3[1][0][2] == 5e+10)
assert (x3[0][0][2] == 5e+8)