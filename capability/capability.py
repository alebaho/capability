# -*- coding: utf-8 -*-
"""
Collecton of utilities.

@author: Alexandre Baharov
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from scipy.stats import norm, chi2
from capability import tables as tbl
import datetime
import math


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
    subsample_means(): list of floats.
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
    confidence(ci=0.95, cal='cpk'): float.
        cal: one of 'cp', 'pp', 'cpk', 'ppk'.
        ci: confidence interval.
        Returns the 100(1-a)% confidence iterval for Cp, Pp, Cpk or Ppk.
    plot():
        Plot a hitogram with limits.

    @author: Alexandre Baharov

    """
    def __init__(self,
                 lst,
                 lsl=None,
                 usl=None,
                 target=None,
                 title=None,
                 unit=None,
                 timestamp=None):
        """Initiliaze attributes to describe a variable."""
        if timestamp == 'now':
            timestamp = datetime.datetime.today().strftime(
                "%b %d, %Y - %H:%M:%S")
        elif timestamp is None:
            timestamp = ''
        self.title = title
        self.unit = unit
        if not isinstance(lst, list):
            raise ValueError("Data must be a list or list of lists.")
        # verify if list of lists or not:
        self.lofl = any(isinstance(el, list) for el in lst)
        self.lst = lst
        if self.lofl == True:
            old_length = len(lst)
            for el, sample in enumerate(lst):
                if len(sample) < 15:
                    del lst[el]
            self.lst = lst
            # print(
            #     f'All samples with lenght less than 15 were deleted.\n\tOld list had {old_length} samples.\n\tThe new list has {len(lst)} samples.'
            # )
        self.m = self.samples()
        self.n = self.subsamples()
        if self.m == 1:
            if sum(self.n) < 5:
                raise ValueError(f'Need more than 5 points!')
        if self.m > 1:
            if self.n[0] < 2:
                raise ValueError("Need at least two values.")
        if lsl is None and usl is None:
            raise ValueError('Need at least one of USL or LSL.')
        self.usl = usl
        self.lsl = lsl
        self.target = target
        try:
            (usl - lsl) / 2
        except ValueError:
            self.midpoint = None
            pass
        else:
            self.midpoint = (usl + lsl) / 2
        self.timestamp = timestamp

    @staticmethod
    def Average(lst):
        """Get average of a list"""
        return sum(lst) / len(lst)

    def samples(self):
        """Count number of samples."""
        if self.lofl == False:
            return 1
        else:
            return sum(1 for el in self.lst)

    def subsamples(self):
        """Count the number of supsamples in a sample."""
        if self.lofl == False:
            return [len(self.lst)]
        else:
            subs_count = []
            for i in range(self.m):
                subs_count.append(len(self.lst[i]))
            return subs_count

    def subsample_means(self):
        """Return the mean for each sample."""
        if self.lofl == False:
            return [Capability.Average(self.lst)]
        else:
            subs_count = []
            for sample in range(self.m):
                subs_count.append(
                    round(Capability.Average(self.lst[sample]), 3))
            return subs_count

    def x_bar(self):
        """Calculate the x-bar."""
        xbar = Capability.Average(self.subsample_means())
        return round(xbar, 3)

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

    def s_long(self):
        """Calculate the long term standard deviation."""
        def stdv(lst):
            """Calculate the standard deviaton"""
            return np.array(lst).std(ddof=1)

        if self.lofl == False:
            if self.n[0] >= 15:
                return round(stdv(self.lst), 7)
            else:
                return round(stdv(self.lst) / tbl.get_c4(self.n[0]), 7)
        else:
            long_list = []
            for sample in range(self.m):
                long_list.extend(self.lst[sample])
            if len(long_list) >= 15:
                return round(stdv(long_list), 7)
            else:
                return round(stdv(long_list) / tbl.get_c4(len(long_list)), 7)

    def s_short(self):
        """Calculate the short term standard deviation."""
        if self.lofl == False:
            return round(self.r_bar() / tbl.get_d2(2), 7)
        else:
            numerator = []
            for sample in range(self.m):
                numerator.append((self.n[sample] - 1) *
                                 (np.array(self.lst[sample]).std(ddof=1))**2)
            v = self.ddof()
            short = math.sqrt(sum(numerator) / v)
            if min(self.n) >= 15:
                return round(short, 7)
            else:
                denominator = tbl.get_c4(sum(self.n)) * (1 + v)
                return round(short / denominator, 7)

    def cp(self):
        """Calculate Cp."""
        if self.lsl is None or self.usl is None:
            raise ValueError("Need both USL and LSL for calculation.")
        return round((self.usl - self.lsl) / (6 * self.s_short()), 3)

    def cpu(self):
        """Calculate Cpu."""
        if self.usl is None:
            raise ValueError('Need USL for calculation!')
        return round((self.usl - self.x_bar()) / (3 * self.s_short()), 3)

    def cpl(self):
        """Calculate Cpl."""
        if self.lsl is None:
            raise ValueError('Need LSL for calculation!')
        return round((self.x_bar() - self.lsl) / (3 * self.s_short()), 3)

    def cpk(self):
        """Calculate Cpk."""
        if self.usl is None:
            return self.cpl()
        elif self.lsl is None:
            return self.cpu()
        else:
            return min(self.cpl(), self.cpu())

    def pp(self):
        """Calculate Pp."""
        if self.lsl is None or self.usl is None:
            raise ValueError("Need both USL and LSL for calculation.")
            return None
        return round((self.usl - self.lsl) / (6 * self.s_long()), 3)

    def ppu(self):
        """Calculate Ppu."""
        if self.usl is None:
            raise ValueError('Need USL for calculation!')
            return None
        return round((self.usl - self.x_bar()) / (3 * self.s_long()), 3)

    def ppl(self):
        """Calculate Ppl."""
        if self.lsl is None:
            raise ValueError('Need LSL for calculation!')
            return None
        return round((self.x_bar() - self.lsl) / (3 * self.s_long()), 3)

    def ppk(self):
        """Calculate Ppk."""
        if self.usl is None:
            return self.ppl()
        elif self.lsl is None:
            return self.ppu()
        else:
            return min(self.ppl(), self.ppu())

    def k(self):
        """Calculate k, the divergence of population mean from target value."""
        if self.usl is None or self.lsl is None:
            raise ValueError('Need both USL and LSL for calculatiion.')
        if self.target is None:
            raise ValueError('Target is missing. Did you forget to enter it?')
        else:
            if self.x_bar() >= self.target:
                _k = (self.x_bar() - self.target) / (self.usl - self.target)
            else:
                _k = (self.x_bar() - self.target) / (self.target - self.lsl)
            return round(_k, 4)

    def zl(self, s='short'):
        """Calculate the lower Z indice."""
        if self.lsl is None:
            raise ValueError("Need LSL for calculation.")
        if s == 'short':
            return round((self.x_bar() - self.lsl) / self.s_short(), 3)
        if s == 'long':
            return round((self.x_bar() - self.lsl) / self.s_long(), 3)

    def zu(self, s='short'):
        """Calculate the upper Z indice."""
        if self.usl is None:
            raise ValueError("Need USL for calculation.")
        if s == 'short':
            return round((self.usl - self.x_bar()) / self.s_short(), 3)
        if s == 'long':
            return round((self.usl - self.x_bar()) / self.s_long(), 3)

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
            non_confm = 1 - norm.cdf(self.zu(s))
        elif self.usl is None:
            non_confm = norm.cdf(-self.zl(s))
        else:
            non_confm = (norm.cdf(-self.zl(s)) + (1 - norm.cdf(self.zu(s))))
        return round(non_confm, 4)

    def dpm(self, s='long'):
        """Calculate DPM Defects Per Milion."""
        _dpm = self.non_conforming(s) * 1000000
        return round(_dpm, 1)

    def yld(self, s='short'):
        """Calculate the capability yield."""
        return round(1 - self.non_conforming(s), 2)

    def sigma_level(self, s='short'):
        """Return the Sigma level."""
        return round(self.z_min(s) + 1.5, 1)

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
        v = self.ddof()
        a = 1 - ci

        if cal == 'cp':
            c = self.cp()
        elif cal == 'pp':
            c = self.pp()
        elif cal == 'cpk':
            c = self.cpk()
        elif cal == 'ppk':
            c = self.ppk()

        if cal == 'cpk' or cal == 'ppk':
            if self.lofl == False:
                n = self.n[0]
            else:
                n = sum(self.n)

            interim = np.sqrt((1 / (9 * n * c**2)) + 1 / (2 * v))

            if self.lsl is None:
                i_low = None
                i_high = round(c * (1 + norm.ppf(1 - a) * interim), 3)
            elif self.usl is None:
                i_low = round(c * (1 - norm.ppf(1 - a) * interim), 3)
                i_high = None
            else:
                i_low = round(c * (1 - norm.ppf(1 - a / 2) * interim), 3)
                i_high = round(c * (1 + norm.ppf(1 - a / 2) * interim), 3)
            return [i_low, i_high]

        elif cal == 'cp' or cal == 'pp':
            if self.lsl is None:
                i_low = None
                i_high = round(c * np.sqrt(chi2.ppf(1 - a, v) / v), 3)
            elif self.usl is None:
                i_low = round(c * np.sqrt(chi2.ppf(a, v) / v), 3)
                i_high = None
            else:
                i_low = round(c * np.sqrt(chi2.ppf(a / 2, v) / v), 3)
                i_high = round(c * np.sqrt(chi2.ppf(1 - a / 2, v) / v), 3)
            return [i_low, i_high]

    def low_interval(self, ci=0.95, cal='cpk'):
        """Return the 100(1-a)% lower confidence."""
        v = self.ddof()
        a = 1 - ci

        if cal == 'cp':
            c = self.cp()
        elif cal == 'pp':
            c = self.pp()
        elif cal == 'cpk':
            c = self.cpk()
        elif cal == 'ppk':
            c = self.ppk()

        if cal == 'cpk' or cal == 'ppk':
            if self.lofl == False:
                n = self.n[0]
            else:
                n = sum(self.n)

            interim = np.sqrt((1 / (9 * n * c**2)) + 1 / (2 * v))
            interval = round(c * (1 - norm.ppf(1 - a) * interim), 3)
            return interval

        elif cal == 'cp' or cal == 'pp':
            interval = round(c * np.sqrt(chi2.ppf(a, v) / v), 3)
            return interval

    def min_sample(self, alpha=0.05, power=0.95, Ha=1.33):
        """Return the minimum sample size."""
        samples = (self.s_long()**2) * (((norm.ppf(power) - norm.ppf(alpha)) /
                                         (self.cpk() - Ha))**2)
        return samples

    def plot(self, kde=True):
        """Plot a hitogram with limits."""
        annotation = f'------------------------------------------------'
        try:
            txt = '\nC$_{{p}}$'
            annotation += f'{txt:<5}{str("="):^3}{self.cp():<7}{self.confidence(cal="cp")}'
        except:
            pass
        txt = '\nC$_{{pk}}$'
        annotation += f'{txt:<5}{str("="):^3}{self.cpk():<7}{self.confidence(cal="cpk")}'
        try:
            txt = '\nP$_{{p}}$'
            annotation += f'{txt:<5}{str("="):^3}{self.pp():<7}{self.confidence(cal="pp")}'
        except:
            pass
        txt = '\nP$_{{pk}}$'
        annotation += f'{txt:<5}{str("="):^3}{self.ppk():<7}{self.confidence(cal="cpk")}'
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
        annotation += f'\nsupsamples = {sub_n}'
        if self.usl is not None:
            annotation += f'\nUSL = {self.usl}'
        if self.lsl is not None:
            annotation += f'\nLSL = {self.lsl}'
        if self.target is not None:
            annotation += f'\nTarget = {self.target}'
        annotation += f'\nMidpoint = {self.midpoint}'
        annotation += f'\n$\overline{{x}}$ = {self.x_bar()}'
        annotation += f'\n$\overline{{R}}$ = {self.r_bar()}'
        annotation += f'\ns$_{{short\ term}}$ = {self.s_short()}'
        annotation += f'\ns$_{{long\ term}}$ = {self.s_long()}'
        try:
            annotation += f'\nk = {self.k()} (div. of mean from target)'
        except:
            pass
        annotation += f'\ndpm$_{{short\ term}}$ = {self.dpm("short"):.0f}'
        annotation += f'\ndpm$_{{long\ term}}$ = {self.dpm("long"):.0f}'
        annotation += f'\nyield$_{{short\ term}}$ = {self.yld("short"):.1%}'
        annotation += f'\nyield$_{{long\ term}}$ = {self.yld("long"):.1%}'
        annotation += f'\nSigmaLevel$_{{short\ term}}$ = {self.sigma_level("short"):.0f}'
        annotation += f'\nSigmaLevel$_{{long\ term}}$ = {self.sigma_level("long"):.0f}'
        if self.timestamp is not None:
            annotation += f'\n\n{self.timestamp}'

        if self.lofl == True:
            s = []
            for sample in range(self.m):
                s.extend(self.lst[sample])
        else:
            s = self.lst

        fig = plt.figure(figsize=(9.5, 5.5),
                         num=self.title,
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
                         stat="density",
                         color='skyblue')
            sns.kdeplot(s, color='k', alpha=0.3)
        elif kde == False:
            sns.histplot(s, bins=bins, element="step", color='skyblue')

        ymax = int(max(counts) + 0.3 * max(counts))
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
        if self.target is not None:
            plt.vlines(x=self.target,
                       ymin=0,
                       ymax=ymax,
                       color='limegreen',
                       linestyle=':',
                       label='Target')
        if self.midpoint is not None:
            plt.vlines(x=self.midpoint,
                       ymin=0,
                       ymax=ymax,
                       color='orange',
                       linestyle='--',
                       label='Midpoint')

        u_3Sigma = 3 * self.s_long() + self.x_bar()
        l_3Sigma = -3 * self.s_long() + self.x_bar()
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
        plt.vlines(x=self.x_bar(),
                   ymin=0,
                   ymax=0.25 * ymax,
                   color='k',
                   linestyle=':',
                   label='x-bar')

        plt.ylim((0, ymax))

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
        plt.xlabel(self.unit)
        plt.title(self.title)
        annotation_box = AnchoredText(annotation,
                                      loc='lower left',
                                      prop=dict(size=8.5),
                                      frameon=False,
                                      bbox_to_anchor=(1., 0.),
                                      bbox_transform=ax.transAxes)
        plt.gca().add_artist(annotation_box)
