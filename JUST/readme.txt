JUST software package written by Ebrahim Ghaderpour
emails: ebrahim.ghaderpour@ucalgary.ca   
        ebrahimg2@gmail.com

To run each script in the MATLAB Command Window first load a time series. 
There are several time series included in this package in .dat format 
whose first, second, and third columns are the times, time series values, and weights.

-------------------------------------------------------------------------------------------

In MATLAB:

TS = load("D:\JUST\EVIA_CA.dat");
t = TS(:,1);     % The time vecor
f = TS(:,2);     % The time series values
P = TS(:,3);     %  Weights

Note: t and f have the same size n. The weights are optional. 
P can also be the inverse of the covariance matrix of order n associated with f.

Then for example call: 
LocIndMagDir = JUSTjumps(t, f, 'P', P, 'save', 1); 
that will pop up a window where you can choose where to save the result as .txt or .dat. 

The jump indices can be extracted as follows: 
JumpInd = LocIndMagDir(:,2);

To display the decomposition results, you may call JUSTdecompose as follows:
JUSTdecompose(t, f, 'P', P, 'ind', JumpInd);

Alternatively, you may call LSWA to decompose the time series into the time-frequency domain:
LSWA(t, f, 'P', P, 'Omega', 0.1:0.1:3.9, 'ind', JumpInd, 'display', 1); 

For regularizing the trend and seasonal components and/or the spectrogram, 
you may first define an equally spaced time vector as follows:
num = floor((t(length(t))-t(1))*52.17);  % 52.17 weeks per year
tt = linspace(t(1), t(length(t)), num); 

Then run
JUSTdecompose(t, f, 'P', P, 'ind', JumpInd, 'tt', tt);
or
LSWA(t, f, 'P', P, 'Omega', 0.1:0.1:3.9, 'ind', JumpInd, 'display', 1, 'tt', tt); 

Users may type help JUST  OR doc JUST in the MATLAB Command Window for the full description 
of the codes and their parameters

-------------------------------------------------------------------------------------------

In Python: 

The Python codes can be run individually and the examples with their descriptions 
are given in the scripts (after if __name__ == '__main__':)

The supported format for the data sets are .csv, .dat, and .txt
Each module imports argparse and can be easily run from the Command Prompt (Fully commented)

Use two consecutive hyphens -- for each input parameter

Example: Try 

D:\JUST>python JUSTdecompose.py --data "D:\JUST\EVIA_CA.csv" --ind 13 111 144 --Numtime 300 --display True --save True

This should display the JUST decomposition results and save the results in a new folder 
called EVIA_CA in the same directory as the input data 

Example: Try

D:\JUST>python ALLSSA.py --datafile "D:\JUST\Sim_Geophysics.dat" --decimal 2 --save True

This will print:
The ALLSSA estimated cyclic frequencies: [4.07, 20.37, 22.29]
And will save the ALLSSA results as .xlsx in a new folder called Sim_Geophysics in the same directory as the input data 

Example:

The input frequencies can be modified by --Lfreq, --Ufreq, and --Numfreq
The input times for regularization in LSWA.py can also be modified by --Ltime, --Utime, and --Numtime
Try:

D:\JUST>python LSWA.py --data "D:\JUST\EVIA_CA.txt" --ind 13 111 144 --Lfreq 0.2 --Ufreq 4.1 --Numfreq 40 --Numtime 300 --display 3

This should display the amplitude spectrogram. The cyclic frequencies are in cycles per year (c/y).
Users may also suppress the annual peaks in the spectrogram (1 c/y) by adding --freq 1 in the command above.
In cases where the matrix of normal equations becomes singular (singularity error), users may change/alter the window size parameters
i.e., by adding --L0 5 to the command as an example 

The full description of parameters is available from Prompt Command using pydoc:

Example:
D:\JUST>python -m pydoc JUSTmonitor
-------------------------------------------------------------------------------------------

Sources of the examples included in this package:

Sim_JumpInd_48_104.dat, ExA_NDVI_Australia.dat, ExB_NDVI_Australia.dat, 
EVIA_CA.dat, EVIB_CA.dat, and EVIC_CA.dat are in 
https://doi.org/10.3390/rs12234001


For near-real-time monitoring, the following three time series are shown in https://doi.org/10.3390/rs12152446

 
You may run JUSTmonitor by inputting the values for 'history' and 'start' as listed below:
  Time series            EVIA_AB.dat    EVIB_AB.dat    EVIC_AB.dat 
   'history'                 18             31             16
    'start'                  73             72             72
Note: These indices are for MATLAB. For Python, 1 should be subtracted from each of them.


sim_Geophysics.dat is the same simulated example in 
https://doi.org/10.1190/geo2017-0284.1
True wavenumbers are 4.0744, 20.3718, 22.2818 achieved by setting 'decimal' to 4 in ALLSSA 


Sim_JumpInd_101_321.dat is the same simulated example in 
https://doi.org/10.1007/s10291-019-0841-3

-------------------------------------------------------------------------------------------

To compute the cross-spectrogam for two time series: (t1, f1, P1) and (t2, f2, P2)
Users may select a common time vector tt for both time series and run LSWA for 
each time series by inputting 'tt' = tt to compute a normalized spectrogram. 
Then multiply the normalized spectrograms entrywise to obtain the cross-spectrogram!
(Reference: https://doi.org/10.1007/s00190-018-1156-9)
Example:
A common time vector tt (equally spaced) for both time series may be obtained by:
a = max (t1(1), t2(1));
b = min (t1(length(t1)), t2(length(t2)));
tt = linspace(a, b, num);  

-------------------------------------------------------------------------------------------

While the software has been tested and debugged using thousands of time series, 
users may still see some bugs for certain time series and inputs. 
I appreciate your feedback that can be posted either on Github or emailed to me. 

Thank you!