import pandas as pd
from collections import OrderedDict

a = "2022/07/21 22/45"

c0_start =  pd.Timestamp(2022, 7, 19, 12, 1) 
c0_end =  pd.Timestamp(2022, 7, 19, 20, 0) 

c1_start =  pd.Timestamp(2022, 7, 20, 7, 50)
c1_end =  pd.Timestamp(2022, 7, 20, 12, 30) 

c2_start =  pd.Timestamp(2022, 7, 20, 22, 28) 
c2_end =  pd.Timestamp(2022, 7, 21, 13, 11)

c3_start =  pd.Timestamp(2022, 7, 21, 15, 6) 
c3_end =  pd.Timestamp(2022, 7, 21, 17, 28) 

c4_start =  pd.Timestamp(2022, 7, 21, 22, 45)
c4_end =  pd.Timestamp(2022, 7, 22, 7, 50) 

c5_start =  pd.Timestamp(2022, 7, 22, 11, 7) 
c5_end =  pd.Timestamp(2022, 7, 22, 16, 0)

c6_start =  pd.Timestamp(2022, 7, 22, 23, 2)
c6_end =  pd.Timestamp(2022, 7, 23, 19, 50) 

c7_start =  pd.Timestamp(2022, 7, 24, 13, 22) 
c7_end =  pd.Timestamp(2022, 7, 24, 19, 33)

closing_times = OrderedDict(
    c0 = (c0_start, c0_end),
    c1 = (c1_start, c1_end),
    c2 = (c2_start, c2_end),
    c3 = (c3_start, c3_end),
    c4 = (c4_start, c4_end),
    c5 = (c5_start, c5_end),
    c6 = (c6_start, c6_end),
    c7 = (c7_start, c7_end),
)