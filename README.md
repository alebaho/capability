# Quality Control
This repository provides tools for calculating cspability indices.

A colection of methods for calculating process capability matrics. 
Uses USL and/or LSL. A minimum condition is that one of USL or LSL is specified.
If the argument of normalize is set to True, the class will verify for normality and if the data is not normal, it uses Yeo-Johnson normalization to normalize the data.
    
Arguments:
---
Argument|Describtion
---|---
lst | The input.
usl | Process uper specification limit.
lsl | Process lower specification limit.
target | Optional pricess target value.
normalize | Whether to normalize or not the data.
alpha | The desried (1-alpha)% confidence interval.


Available Methods
---
Method|Describtion
---|---
usl | The USL if entered in function call.   
lss | The LSL if entered in function call.   
target | The target if entered in function call.   
midpoint | The midpoint between USL and LSL.   
m | The number of sample groups.   
n | The number of supsamples in each sample group.   
subsample_means | Return the n means for each sample group m.   
x_bar | Calculate the x-bar.   
r_bar | Calculate the r-bar.   
s_long | Calculate the long term standard deviation.   
s_short | Calculate the short term standard deviation.   
cp | Calculate Cp. Needs USL and LSL to be specified..   
cpu | Calculate Cpu. Needs USL to be specified.   
cpl| Calculate Cpl. Needs LSL to be specified.   
cpk | Calculate Cpk. Needs USL or LSL to be specified.   
pp | Calculate Pp. Needs USL and LSL to be specified.   
ppu | Calculate Ppu. Needs USL to be specified.   
ppl | Calculate Ppl. Needs LSL to be specified.   
ppk | Calculate Ppk. Needs USL or LSL to be specified.   
k | Calculates divergence of population mean from target value. USL, LSL and Target must be specified.   
zl() | Calculate the lower Z indice for short or long standard deviation. Needs LSL to be specified.   
zu() | Calculate the upper Z indice for short or long standard deviation. Needs USL to be specified.   
z_min() | Return the minimal Z value for short or long standard deviation. Needs USL or LSL to be specified.   
non_conforming() | Calculate the proportion of non-conformings for short or long standard deviation. Needs USL or LSL to be specified.   
dpm() | Calculate DPM Defects Per Milion for short or long standard deviation. Needs USL or LSL to be specified.   
yld() | Calculate the capability yield for short or long standard deviation. Needs USL or LSL to be specified.   
sigma_level() | Return the Sigma level for short or long standard deviation. Needs USL or LSL to be specified.   
ddof | Calculate the degrees of freedom.   
confidence() | Returns the 100(1-a)% confidence iterval for Cp, Pp, Cpk or Ppk.   
low_interval() | Returns the 100(1-a)% lower confidence.   
n_min() | Find the minimum sample size requred to meet desired alpha level and relative error level.   
plot() | Plot a hitogram with limits.   
normal_test() | Verify if distibution is normal using Shapiro.   
\_make_normal()| Creates a new class witth normalized data using Yeo-Johnson.   
Average() | Return the average of a list.   
Make_Long() | Flatten the list of lists into a single list.   
Brake_Down(n, rand, rand_j) | Brake down a list into sublists of size n. If n is a list, each sublist will be of lenght equal to each element of the list n. If rand is True, each sublist has lenght of n +/- rand_j with the last sublist containing the unused items of lst.   
samples() | Count number of samples.   
subsamples() | Count the number of subsamples in a sample.   
subsample_means() | Return the mean for each sample.   
