# -*- coding: utf-8 -*-
"""
Collecton of utilities.

@author: Alexandre Baharov
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from scipy import stats
from capability import tables as tbl
import math
import statsmodels.api as sm


class Capability():
    """
    A colection of methods for calculating process capability matrics.

    The methods are capability ratios, performance ratios, confidence
    interval, short term and long term standard deviations, USL, LSL, Target,
    number of samples, number of supsamples, x-bar, subsample means, r-bar, k,
    z levels, non-conformities, defects per milion, yield, sigma level, degrees
    of freedom.

    Uses USL and/or LSL. A minimum condition is that one of USL or LSL is
    specified.

    Args:
    -------
    lst: list or array, mandatory
        The input.
    usl: int or float, optional. Default is None.
        process uper specification limit.
    lsl: int or float, optional. Default is None.
        Process lower specification limit.
    target: int of float, optional. Default is None.

    Methods
    -------
    usl: float
        The USL if entered in function call.
    lsl: float
        The LSL if entered in function call.
    target: float
        The target if entered in function call.
    midpoint: float
        The midpoint between USL and LSL.
    m: float
        The number of sample groups.
    n: float
        The number of supsamples in each sample group.
    subsample_means: list of floats.
        Return the n means for each sample group m.
    x_bar(): float.
        Calculate the x-bar.
    r_bar(): float.
        Calculate the r-bar.
    s_long(): float.
        Calculate the long term standard deviation.
    s_short(): float.
        Calculate the short term standard deviation.
    cp(): float.
        Calculate Cp. Needs USL and LSL to be specified..
    cpu(): float.
        Calculate Cpu. Needs USL to be specified.
    cpl(): float.
        Calculate Cpl. Needs LSL to be specified.
    cpk(): float.
        Calculate Cpk. Needs USL or LSL to be specified.
    pp(): float.
        Calculate Pp. Needs USL and LSL to be specified..
    ppu(): float.
        Calculate Ppu. Needs USL to be specified.
    ppl(): float.
        Calculate Ppl. Needs LSL to be specified.
    ppk(): float.
        Calculate Ppk. Needs USL or LSL to be specified.
    k(): float.
        Calculates divergence of population mean from target value.
        USL, LSL and Target must be specified.
    zl(s='short'): float.
        Options: 'short', 'long'.
        Calculate the lower Z indice for short or long standard deviation.
        Needs LSL to be specified.
    zu(s='short'): float.
        Options: 'short', 'long'.
        Calculate the upper Z indice for short or long standard deviation.
        Needs USL to be specified.
    z_min(s='short'): float.
        Options: 'short', 'long'.
        Return the minimal Z value for short or long standard deviation.
        Needs USL or LSL to be specified.
    non_conforming(s='long'): float.
        Options: 'short', 'long'.
        Calculate the proportion of non-conformings for short or long
        standard deviation. Needs USL or LSL to be specified.
    dpm(s='long'): float.
        Options: 'short', 'long'.
        Calculate DPM Defects Per Milion for short or long standard
        deviation. Needs USL or LSL to be specified.
    yld(s='short'): float.
        Options: 'short', 'long'.
        Calculate the capability yield for short or long standard
        deviation. Needs USL or LSL to be specified.
    sigma_level(s='short'): float.
        Options: 'short', 'long'.
        Return the Sigma level for short or long standard deviation.
        Needs USL or LSL to be specified.
    ddof():
        Calculate the degrees of freedom.
    confidence(ci=0.95, cal='cpk'): list of float.
        Returns the 100(1-a)% confidence iterval for Cp, Pp, Cpk or Ppk.
        cal: one of 'cp', 'pp', 'cpk', 'ppk'.
        ci: confidence interval.
    low_interval(ci=0.95, cal='cpk'): float
        Returns the 100(1-a)% lower confidence.
        cal: one of 'cp', 'pp', 'cpk', 'ppk'.
        ci: confidence interval.
    n_min(cpk_eval=None, alpha=0.05, err=.1, sides='one'): int
        Find the minimum sample size requred to meet desired alpha level
        and relative error level.
        cpk_eval: The level at which to evaluate the requried nuber of samples. If none provided, the Cpk for 'lst' will be used.
        alpha: the desired Type-I error.
        err: the relative error from .01 to 1.
        sides: one of 'one' or 'two'. The one sided or two sided evaluation. It two sided evaluatin, alpha is devided by two.
    plot():
        Plot a hitogram with limits.
    normal_test(lst, alpha=0.05): boolean
        Verify if distibution is normal.
        lst: a list to be tested for normality usig Shapiro.
        alpha: the alpha confidence level for the Shapiro test.
    _make_normal(self): class Capability
        Creates a new class witth normalized data using Yeo-Johnson.
    Average(lst): float
        Return the average of a list.
        lst: a list to be averaged.
    Make_Long(lists): list
        Flatten the list of lists into a single list.
        lists: the lists to be put into a single list.
    Brake_Down(lst, n, rand=False, rand_j=0): list of lists
        Brake down a list into sublists of size n.
        If n is a list, each sublist will be of lenght equal to each element of the list n.
        If rand is True, each sublist has lenght of n +/- rand_j with the
        last sublist containing the unused items of lst.
        rand and rand_j will be disregarded if n is a list.
        lst: a list to be broken down into smaller lists.
        n: an integer or a list of integers for each lenght of the sublists.
        rand: boolen. If the lenght of each sublist is to be variable.
        rand_j: if rand==True, the +/- value to be added to n to create variable sublists of size n+/-rand_j.
    samples(self): int
        Count number of samples.
    subsamples(self): list
        Count the number of subsamples in a sample.
    subsample_means(self): float
        Return the mean for each sample.

    @author: Alexandre Baharov

    """
    def __init__(self,
                 lst,
                 lsl=None,
                 usl=None,
                 target=None,
                 normalise=True,
                 alpha=0.05):
        """Initiliaze attributes to describe a variable."""
        self.lst = lst
        self.usl = usl
        self.lsl = lsl
        self.target = target
        self.alpha = alpha

        if not isinstance(lst, list):
            raise ValueError("Data must be a list or list of lists.")

        # verify if list of lists or not:
        self.lofl = any(isinstance(sublist, list) for sublist in lst)

        if self.lofl == True:
            old_length = len(lst)
            for idx, sample in enumerate(lst):
                if len(sample) < 5:
                    del lst[idx]
            self.lst = lst
            if len(lst) < old_length:
                print(
                    f'All samples with lenght less than 15 were deleted.\n\tOld list had {old_length} samples.\n\tThe new list has {len(lst)} samples.'
                )
        self.m = self.samples
        self.n = self.subsamples

        if self.lofl == False:
            if sum(self.n) < 5:
                raise ValueError(f'Need more than 5 points!')
        if self.lofl == True:
            if self.n[0] < 2:
                raise ValueError("Need at least two values.")
        if lsl is None and usl is None:
            raise ValueError('Need at least one of USL or LSL.')
        try:
            (usl - lsl) / 2
        except ValueError:
            self.midpoint = None
            pass
        else:
            self.midpoint = (usl + lsl) / 2
            if self.target is None:
                self.target = self.midpoint

        if normalise == True:
            self._make_normal()

    @staticmethod
    def normal_test(lst, alpha=0.05):
        """Verify if distibution is normal."""
        lst = np.array(lst)
        _, p = stats.shapiro(lst)
        if p < alpha:
            print(f'Data is not normaly distributed.')
            return False
        else:
            print(f'Distribution is normal.')
            return True

    def _make_normal(self):
        """Normalize a list using Yeo-Johnson."""
        if self.lofl == False:
            is_normal = self.normal_test(self.lst, self.alpha)
            if is_normal == False:
                lst, self.lmbda = stats.yeojohnson(self.lst)
                self.lst = lst.tolist()
            else:
                return
        if self.lofl == True:
            lst, self.lmbda = stats.yeojohnson(self.Make_Long(self.lst))
            lst = lst.tolist()
            self.lst = Capability.Brake_Down(lst=lst, n=self.n)
            # normalised_lofl = []
            # for sublist in self.lst:
            #     sublist, _ = stats.yeojohnson(sublist)
            #     normalised_lofl.append(list(sublist))
            # self.lst = normalised_lofl

        if self.lsl is not None:
            lsl = stats.yeojohnson(self.lsl, self.lmbda).tolist()
            lsl = round(lsl, 3)
        if self.usl is not None:
            usl = stats.yeojohnson(self.usl, self.lmbda).tolist()
            usl = round(usl, 3)
        self.lsl = min(lsl, usl)
        self.usl = max(lsl, usl)
        if self.target is not None:
            self.target = stats.yeojohnson(self.target, self.lmbda).tolist()
            self.target = round(self.target, 3)
        try:
            (self.usl - self.lsl) / 2
        except ValueError:
            self.midpoint = None
            pass
        else:
            self.midpoint = (self.usl + self.lsl) / 2
            self.midpoint = round(self.midpoint, 3)

    @staticmethod
    def Average(lst):
        """Return the average of a list."""
        return sum(lst) / len(lst)

    @staticmethod
    def Make_Long(lists):
        """Flatten the list of lists into a single list."""
        return [item for sublist in lists for item in sublist]

    @staticmethod
    def Brake_Down(lst, n, rand=False, rand_j=0):
        """Brake down a list into sublists of size n.
            If n is a list, each sublist will be of lenght equal to each element of the list n.
            If rand is True, each sublist has lenght of n +/- rand_j with the
            last sublist containing the unused items of lst.
            rand and rand_j will be disregarded if n is a list."""
        max = len(lst)
        l4 = []
        i = 0
        j = 0
        while max > 0:
            if isinstance(n, list):
                k = n[j]
                j += 1
            elif rand == False:
                k = n
            else:
                k = np.random.randint(n - rand_j, n + rand_j, 1).tolist()[0]
            l4.append(lst[i:i + k])
            max -= k
            i += k
        return l4

    @property
    def samples(self):
        """Count number of samples."""
        if self.lofl == False:
            return 1
        else:
            return sum(1 for sublist in self.lst)

    @property
    def subsamples(self):
        """Count the number of subsamples in a sample."""
        if self.lofl == False:
            return [len(self.lst)]
        else:
            return [len(sublist) for sublist in self.lst]

    @property
    def subsample_means(self):
        """Return the mean for each sample."""
        if self.lofl == False:
            return [Capability.Average(self.lst)]
        else:
            return [Capability.Average(sublist) for sublist in self.lst]

    @property
    def x_bar(self):
        """Calculate the x-bar."""
        xbar = Capability.Average(self.subsample_means)
        return round(xbar, 3)

    @property
    def r_bar(self):
        """Calculate the r-bar."""
        r = []
        if self.lofl == False:
            for subs in range(self.n[0] - 1):
                r.append(abs(self.lst[subs + 1] - self.lst[subs]))
        else:
            for subs in range(self.m):
                r.append(max(self.lst[subs]) - min(self.lst[subs]))
        return round(Capability.Average(r), 3)

    @property
    def s_long(self):
        """Calculate the long term standard deviation."""
        def stdv(lst):
            """Calculate the standard deviaton"""
            return np.array(lst).std(ddof=1)

        if self.lofl == False:
            if self.n[0] >= 5:
                return round(stdv(self.lst), 7)
            else:
                return round(stdv(self.lst) / tbl.get_c4(self.n[0]), 7)
        else:
            long_list = Capability.Make_Long(self.lst)
            if len(long_list) >= 5:
                return round(stdv(long_list), 7)
            else:
                return round(stdv(long_list) / tbl.get_c4(len(long_list)), 7)

    @property
    def s_short(self):
        """Calculate the short term standard deviation."""
        if self.lofl == False:
            return round(self.r_bar / tbl.get_d2(2), 7)
        else:
            numerator = []
            for sample in range(self.m):
                numerator.append((self.n[sample] - 1) *
                                 (np.array(self.lst[sample]).std(ddof=1))**2)
            v = self.ddof
            short = math.sqrt(sum(numerator) / v)
            if min(self.n) >= 5:
                return round(short, 7)
            else:
                denominator = tbl.get_c4(1 + v)
                return round(short / denominator, 7)

    @property
    def cp(self):
        """Calculate Cp."""
        if self.lsl is None or self.usl is None:
            raise ValueError("Need both USL and LSL for calculation.")
        return round((self.usl - self.lsl) / (6 * self.s_short), 3)

    @property
    def cpu(self):
        """Calculate Cpu."""
        if self.usl is None:
            raise ValueError('Need USL for calculation!')
        return round((self.usl - self.x_bar) / (3 * self.s_short), 3)

    @property
    def cpl(self):
        """Calculate Cpl."""
        if self.lsl is None:
            raise ValueError('Need LSL for calculation!')
        return round((self.x_bar - self.lsl) / (3 * self.s_short), 3)

    @property
    def cpk(self):
        """Calculate Cpk."""
        if self.usl is None:
            return self.cpl
        elif self.lsl is None:
            return self.cpu
        else:
            return min(self.cpl, self.cpu)

    @property
    def cpks(self):
        """Calculater the list of cpk for each sublist in a list"""
        if not self.lofl == True:
            raise ValueError("Need at least a list with two sublists.")
        else:
            return [
                Capability(sublist, usl=self.usl, lsl=self.lsl).cpk
                for sublist in self.lst
            ]

    @property
    def pp(self):
        """Calculate Pp."""
        if self.lsl is None or self.usl is None:
            raise ValueError("Need both USL and LSL for calculation.")
            return None
        return round((self.usl - self.lsl) / (6 * self.s_long), 3)

    @property
    def ppu(self):
        """Calculate Ppu."""
        if self.usl is None:
            raise ValueError('Need USL for calculation!')
            return None
        return round((self.usl - self.x_bar) / (3 * self.s_long), 3)

    @property
    def ppl(self):
        """Calculate Ppl."""
        if self.lsl is None:
            raise ValueError('Need LSL for calculation!')
            return None
        return round((self.x_bar - self.lsl) / (3 * self.s_long), 3)

    @property
    def ppk(self):
        """Calculate Ppk."""
        if self.usl is None:
            return self.ppl
        elif self.lsl is None:
            return self.ppu
        else:
            return min(self.ppl, self.ppu)

    @property
    def k(self):
        """Calculate k, the divergence of population mean from target value."""
        if self.usl is None or self.lsl is None:
            raise ValueError('Need both USL and LSL for calculatiion.')
        if self.target is None:
            raise ValueError('Target is missing. Did you forget to enter it?')
        else:
            if self.x_bar >= self.target:
                _k = (self.x_bar - self.target) / (self.usl - self.target)
            else:
                _k = (self.x_bar - self.target) / (self.target - self.lsl)
            return round(_k, 4)

    def zl(self, s='short'):
        """Calculate the lower Z indice."""
        if self.lsl is None:
            raise ValueError("Need LSL for calculation.")
        if s == 'short':
            return round((self.x_bar - self.lsl) / self.s_short, 3)
        if s == 'long':
            return round((self.x_bar - self.lsl) / self.s_long, 3)

    def zu(self, s='short'):
        """Calculate the upper Z indice."""
        if self.usl is None:
            raise ValueError("Need USL for calculation.")
        if s == 'short':
            return round((self.usl - self.x_bar) / self.s_short, 3)
        if s == 'long':
            return round((self.usl - self.x_bar) / self.s_long, 3)

    def z_min(self, s='short'):
        """Return the minimal Z value."""
        if self.usl is None:
            return self.zl(s)
        elif self.lsl is None:
            return self.zu(s)
        else:
            return min(self.zu(s), self.zl(s))

    def non_conforming(self, s='long'):
        """Calculate the proportion of non-conformings."""
        if self.lsl is None:
            non_confm = 1 - stats.norm.cdf(self.zu(s))
        elif self.usl is None:
            non_confm = stats.norm.cdf(-self.zl(s))
        else:
            non_confm = (stats.norm.cdf(-self.zl(s)) +
                         (1 - stats.norm.cdf(self.zu(s))))
        return round(non_confm, 4)

    def dpm(self, s='long'):
        """Calculate DPM Defects Per Milion."""
        _dpm = self.non_conforming(s) * 1000000
        return int(round(_dpm, 0))

    def yld(self, s='short'):
        """Calculate the capability yield."""
        return round(1 - self.non_conforming(s), 2)

    def sigma_level(self, s='short'):
        """Return the Sigma level."""
        return round(self.z_min(s) + 1.5, 1)

    @property
    def ddof(self):
        """Calculate the degrees of freedom."""
        if self.lofl == False:
            if self.n[0] == 1:
                raise ValueError('At least two data points are needed!')
            else:
                return self.n[0] - 1
        else:
            v = []
            for sample in range(self.m):
                v.append(self.n[sample] - 1)
            return sum(v)

    def confidence(self, ci=0.95, cal='cpk'):
        """Return the 100(1-a)% confidence iterval for Cp, Pp, Cpk or Ppk."""
        v = self.ddof
        a = 1 - ci

        if cal == 'cp':
            c = self.cp
        elif cal == 'pp':
            c = self.pp
        elif cal == 'cpk':
            c = self.cpk
        elif cal == 'ppk':
            c = self.ppk

        if cal == 'cpk' or cal == 'ppk':
            if self.lofl == False:
                n = self.n[0]
            else:
                n = sum(self.n)

            interim = np.sqrt((1 / (9 * n * c**2)) + 1 / (2 * v))

            if self.lsl is None:
                i_low = None
                i_high = round(c * (1 + stats.norm.ppf(1 - a) * interim), 3)
            elif self.usl is None:
                i_low = round(c * (1 - stats.norm.ppf(1 - a) * interim), 3)
                i_high = None
            else:
                i_low = round(c * (1 - stats.norm.ppf(1 - a / 2) * interim), 3)
                i_high = round(c * (1 + stats.norm.ppf(1 - a / 2) * interim),
                               3)
            low = min(i_low, i_high)
            high = max(i_low, i_high)
            return [low, high]

        elif cal == 'cp' or cal == 'pp':
            if self.lsl is None:
                i_low = None
                i_high = round(c * np.sqrt(stats.chi2.ppf(1 - a, v) / v), 3)
            elif self.usl is None:
                i_low = round(c * np.sqrt(stats.chi2.ppf(a, v) / v), 3)
                i_high = None
            else:
                i_low = round(c * np.sqrt(stats.chi2.ppf(a / 2, v) / v), 3)
                i_high = round(c * np.sqrt(stats.chi2.ppf(1 - a / 2, v) / v),
                               3)
            low = min(i_low, i_high)
            high = max(i_low, i_high)
            return [low, high]

    def low_interval(self, ci=0.95, cal='cpk'):
        """Return the 100(1-a)% lower confidence."""
        v = self.ddof
        a = 1 - ci

        if cal == 'cp':
            c = self.cp
        elif cal == 'pp':
            c = self.pp
        elif cal == 'cpk':
            c = self.cpk
        elif cal == 'ppk':
            c = self.ppk

        if cal == 'cpk' or cal == 'ppk':
            if self.lofl == False:
                n = self.n[0]
            else:
                n = sum(self.n)

            interim = np.sqrt((1 / (9 * n * c**2)) + 1 / (2 * v))
            interval = round(c * (1 - stats.norm.ppf(1 - a) * interim), 3)
            return interval

        elif cal == 'cp' or cal == 'pp':
            interval = round(c * np.sqrt(stats.chi2.ppf(a, v) / v), 3)
            return interval

    def n_min(self, cpk_eval=None, err=.1, sides='one'):
        """Find the minimum sample size"""
        def n_sub_min(lists):
            """Return the minimum sample for each list in a list of lists"""
            return max([
                Capability(sublist,
                           lsl=self.lsl,
                           usl=self.usl,
                           normalise=False).n_min(err=err) for sublist in lists
            ])

        def calculate_n(cpk_eval, alpha, err):
            """Calculate n for a given Cpk level."""
            if cpk_eval == None:
                cpk_eval = self.cpk
            n = 2
            rhs = (3 * (err) / norm.ppf(alpha))**2
            lhs = 1 / (n * cpk_eval**2) + 9 / (2 * n - 2)
            while lhs >= rhs:
                n += 1
                lhs = 1 / (n * cpk_eval**2) + 9 / (2 * n - 2)
            return int(n)

        if sides == 'two':
            alpha = self.alpha / 2
        else:
            alpha = self.alpha

        if self.lofl == False:
            n = calculate_n(cpk_eval=cpk_eval, alpha=alpha, err=err)
        else:
            l_long = self.Make_Long(self.lst)
            n_long = Capability(l_long,
                                usl=self.usl,
                                lsl=self.lsl,
                                normalise=False).n_min(err=err)
            l4 = Capability.brake_down(lst=l_long, n=n_long)
            n1 = n_sub_min(l4)
            l_temp = Capability.brake_down(lst=l_long, n=n1)
            n2 = n_sub_min(l_temp)
            n = max(n1, n2)
        return n

    def plot(self, kde=True, title=None, unit=None, stat='density'):
        """Plot a hitogram with limits."""
        annotation = f'------------------------------------------------'
        try:
            txt = '\nC$_{{p}}$'
            annotation += f'{txt:<5}{str("="):^3}{self.cp:<7}{self.confidence(cal="cp")}{1-self.alpha:.0%}'
        except:
            pass
        txt = '\nC$_{{pk}}$'
        annotation += f'{txt:<5}{str("="):^3}{self.cpk:<7}{self.confidence(cal="cpk")}{1-self.alpha:.0%}'
        try:
            txt = '\nP$_{{p}}$'
            annotation += f'{txt:<5}{str("="):^3}{self.pp:<7}{self.confidence(cal="pp")}{1-self.alpha:.0%}'
        except:
            pass
        txt = '\nP$_{{pk}}$'
        annotation += f'{txt:<5}{str("="):^3}{self.ppk:<7}{self.confidence(cal="ppk")}{1-self.alpha:.0%}'
        annotation += f'\n-----------------------------------------------'
        annotation += f'\ntotal points  = {sum(self.n)}'
        annotation += f'\nsamples = {self.m}'
        if self.m == 1:
            sub_n = self.n[0]
        else:
            sub_n = sum(self.n) / self.n[0]
            if sub_n == len(self.n):
                sub_n = self.n[0]
            else:
                sub_n = f'Variable subsamles.\n                        From {min(self.n)} to {max(self.n)}'
        annotation += f'\nsubsamples = {sub_n}'
        if self.usl is not None:
            annotation += f'\nUSL = {self.usl}'
        if self.lsl is not None:
            annotation += f'\nLSL = {self.lsl}'
        if self.target is not None:
            annotation += f'\nTarget = {self.target}'
        annotation += f'\nMidpoint = {self.midpoint}'
        annotation += f'\n$\overline{{x}}$ = {self.x_bar}'
        annotation += f'\n$\overline{{R}}$ = {self.r_bar}'
        annotation += f'\ns$_{{short\ term}}$ = {self.s_short}'
        annotation += f'\ns$_{{long\ term}}$ = {self.s_long}'
        try:
            annotation += f'\nk = {self.k} (div. of mean from target)'
        except:
            pass
        annotation += f'\ndpm$_{{short\ term}}$ = {self.dpm("short"):.0f}'
        annotation += f'\ndpm$_{{long\ term}}$ = {self.dpm("long"):.0f}'
        annotation += f'\nyield$_{{short\ term}}$ = {self.yld("short"):.1%}'
        annotation += f'\nyield$_{{long\ term}}$ = {self.yld("long"):.1%}'
        annotation += f'\nSigmaLevel$_{{short\ term}}$ = {self.sigma_level("short"):.0f}'
        annotation += f'\nSigmaLevel$_{{long\ term}}$ = {self.sigma_level("long"):.0f}'
        try:
            annotation += f'\nlambda = {self.lmbda:.3f}'
        except:
            pass
        annotation += f'\nn$_{{min}}$ >= {self.n_min()}  rel.error = 10%'

        if self.lofl == True:
            s = Capability.Make_Long(self.lst)
        else:
            s = self.lst

        fig = plt.figure(figsize=(9.5, 5.5),
                         num=title,
                         frameon=False,
                         constrained_layout=True)
        ax = fig.add_subplot()

        right_side = ax.spines["right"]
        right_side.set_visible(False)
        left_side = ax.spines["left"]
        left_side.set_visible(False)
        top_side = ax.spines["top"]
        top_side.set_visible(False)

        counts, bins = np.histogram(s, bins='auto', density=True)

        if kde == True:
            sns.histplot(s,
                         bins=bins,
                         element="step",
                         stat=stat,
                         color='skyblue').set(ylim=(0))
            sns.kdeplot(s, color='k', alpha=0.3).set(ylim=(0))
        elif kde == False:
            sns.histplot(s,
                         bins=bins,
                         element="step",
                         color='skyblue',
                         stat=stat).set(ylim=(0))

        ymax = max(counts) + 0.3 * max(counts)

        if self.lsl is not None:
            plt.vlines(x=self.lsl,
                       ymin=0,
                       ymax=ymax,
                       color='deeppink',
                       label='LSL')
        if self.usl is not None:
            plt.vlines(x=self.usl,
                       ymin=0,
                       ymax=ymax,
                       color='deeppink',
                       label='USL')
        if self.midpoint is not None:
            plt.vlines(x=self.midpoint,
                       ymin=0,
                       ymax=ymax,
                       color='deeppink',
                       linestyle='--',
                       label='Midpoint')
        if self.target is not None:
            plt.vlines(x=self.target,
                       ymin=0,
                       ymax=ymax,
                       color='limegreen',
                       linestyle=':',
                       label='Target')

        u_3Sigma = 3 * self.s_long + self.x_bar
        l_3Sigma = -3 * self.s_long + self.x_bar
        plt.vlines(x=u_3Sigma,
                   ymin=0,
                   ymax=0.15 * ymax,
                   color='r',
                   linestyle=':',
                   label='+3 sigma')
        plt.vlines(x=l_3Sigma,
                   ymin=0,
                   ymax=0.15 * ymax,
                   color='r',
                   linestyle=':',
                   label='-3 sigma')
        plt.vlines(x=self.x_bar,
                   ymin=0,
                   ymax=0.25 * ymax,
                   color='k',
                   linestyle=':',
                   label='x-bar')

        plt.ylim(bottom=0, top=ymax)

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(
            framealpha=1,
            edgecolor='white',
            fontsize=8,
            # frameon=False,
            bbox_to_anchor=(0.05, 1),
            loc='upper left')
        plt.xlabel(unit)
        plt.title(title)
        annotation_box = AnchoredText(annotation,
                                      loc='lower left',
                                      prop=dict(size=8.5),
                                      frameon=False,
                                      bbox_to_anchor=(1., 0.),
                                      bbox_transform=ax.transAxes)
        plt.gca().add_artist(annotation_box)
        plt.show()