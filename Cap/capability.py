# -*- coding: utf-8 -*-
"""
Collecton of utilities.

@author: Alexandre Baharov
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import norm, chi2
import tables as tbl_cap
import datetime


class Capability():
    """
    A colection of methods for calculating process capability matrics.

    The methods are capability ratios, performance ratios, confidence
    interval, short term and long term standard deviations, USL, LSL, Target,
    number of samples, number of subsamples, x-bar, subsample means, r-bar, k,
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
        The number of subsamples in each sample group.
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
    calculate_ddof():
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
                "%b %d, %Y\n%H:%M:%S")
        elif timestamp is None:
            timestamp = ''
        if not isinstance(lst, (np.ndarray, list)):
            raise ValueError("Data must be a list or ndarray type.")
        self.lst = lst
        self.m = self.samples()
        self.n = self.subsamples()
        if self.m == 1 and self.n < 2:
            raise ValueError("Need at least two values.")
        if self.m > 1 and self.n[0] < 2:
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
        self.title = title
        self.unit = unit
        self.timestamp = timestamp

    def samples(self):
        """Count number of samples."""
        ary = np.array(self.lst)
        if len(ary.shape) == 1:
            return 1
        else:
            return ary.shape[0]

    def subsamples(self):
        """Count the number of subsamples in a sample."""
        ary = np.array(self.lst)
        if len(ary.shape) == 1:
            return ary.shape[0]
        else:
            subsample_count = []
            for i in range(ary.shape[0]):
                subsample_count.append(ary[i].shape[0])
        return subsample_count

    def subsample_means(self):
        """Return the mean for each sample."""
        ary = np.array(self.lst)
        if len(ary.shape) == 1:
            return self.x_bar()
        else:
            subsample_means = []
            for i in range(ary.shape[0]):
                subsample_means.append(round(np.mean(ary[i]), 3))
        return subsample_means

    def x_bar(self):
        """Calculate the x-bar."""
        if self.m == 1:
            xbar = np.average(self.lst)
        else:
            x_means = []
            for i in range(self.m):
                x_means.append(np.mean(self.lst[i]))
            xbar = np.average(x_means)
        return round(xbar, 3)

    def r_bar(self):
        """Calculate the r-bar."""
        r = []
        if self.m == 1:
            for i in range(len(self.lst) - 1):
                r.append(abs(self.lst[i + 1] - self.lst[i]))
        else:
            for sample in range(self.m):
                r.append(max(self.lst[sample]) - min(self.lst[sample]))
        return round(np.mean(r), 3)

    def s_long(self):
        """Calculate the long term standard deviation."""
        if self.m == 1 and self.n >= 15:
            return round(np.array(self.lst).std(ddof=1), 7)
        elif self.m == 1 and self.n < 15:
            return round(
                np.array(self.lst).std(ddof=1) / tbl_cap.get_c4(self.n), 7)
        else:
            s = []
            for i in range(self.m):
                s.extend(self.lst[i])
            if len(s) >= 15:
                return round(np.array(s).std(ddof=1), 7)
            else:
                return round(
                    np.array(s).std(ddof=1) / tbl_cap.get_c4(len(s)), 7)

    def s_short(self):
        """Calculate the short term standard deviation."""
        if self.m == 1:
            return round(self.r_bar() / tbl_cap.get_d2(2), 7)
        elif self.m > 1 and sum(self.n) >= 15:
            s_temp = []
            for i in range(self.m):
                s_temp.append(self.n[i] * np.array(self.lst[i]).std(ddof=1))
            return round(sum(s_temp) / sum(self.n), 7)
        else:
            s_temp_nom = []
            s_temp_denom = []
            for i in range(self.m):
                temp = 1 - tbl_cap.get_c4(self.n[i]) * (self.n[i]**2)
                s = np.array(self.lst[i]).std(ddof=1)
                nominator = (
                    (tbl_cap.get_c4(self.n[i]) * self.n[i]) / temp) * s
                s_temp_nom.append(nominator)
                denominator = ((tbl_cap.get_c4(self.n[i]) * (self.n[i]**2)) /
                               temp)
                s_temp_denom.append(denominator)
            return round(sum(s_temp_nom) / sum(s_temp_denom), 7)

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
        """Calculate K. Divergence of population mean from target value."""
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
        return round(100 * (1 - self.non_conforming(s)), 2)

    def sigma_level(self, s='short'):
        """Return the Sigma level."""
        return round(self.z_min(s) + 1.5, 1)

    def calculate_ddof(self):
        """Calculate the degrees of freedom."""
        if self.n == 1:
            raise ValueError('At least two data points are needed!')
        elif self.m == 1 and self.n < 15:
            return 0.88 * (self.n - 1)
        elif self.m == 1 and self.n >= 15:
            return self.n - 1
        else:
            s = []
            for i in range(self.m):
                s.extend(self.lst[i])
            if len(s) >= 15:
                return len(s) - 1
            else:
                return 0.88 * (len(s) - 1)

    def confidence(self, ci=0.95, cal='cpk'):
        """Return the 100(1-a)% confidence iterval for Cp, Pp, Cpk or Ppk."""
        v = self.calculate_ddof()
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
            if self.m == 1:
                n = self.n
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

    def plot(self, kde=True):
        """Plot a hitogram with limits."""
        annotation = f'n  = {self.m*self.n}\n'
        annotation += f'samples = {self.m}\n'
        annotation += f'subsamples = {self.n}\n'
        annotation += f'$\overline{{x}}$ = {self.x_bar()}\n'
        annotation += f'$\overline{{R}}$ = {self.r_bar()}\n'
        annotation += f'$s_{{short}}$ = {self.s_short()}\n'
        annotation += f'$s_{{long}}$ = {self.s_long()}\n'
        if self.lsl is not None and self.usl is not None:
            annotation += f'$C_{{p}}$ = {self.cp()}\n'
            annotation += f'$P_{{p}}$ = {self.pp()}\n'
        # annotation += f'$C_{{pk}}$ = {self.cpk()}\n'
        # annotation += f'$P_{{pk}}$ = {self.ppk()}'
        if self.timestamp is not None:
            annotation += f'\n\n{self.timestamp}'

        if self.m > 1:
            s = []
            for i in range(self.m):
                s.extend(self.lst[i])
        else:
            s = self.lst

        # fig = plt.figure()
        ax = plt.subplot(111)

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

        # Shrink current axis by 10%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        plt.legend(framealpha=1,
                   fontsize='small',
                   frameon=False,
                   bbox_to_anchor=(1.005, 1),
                   loc='upper left')
        plt.xlabel(self.unit)
        plt.title(self.title)
        annotation_box = mpl.offsetbox.AnchoredText(annotation,
                                                    prop=dict(size=8),
                                                    frameon=False,
                                                    bbox_to_anchor=(1.005, 0),
                                                    loc='lower left')
        annotation_box.patch.set_boxstyle("round, pad=0., rounding_size=0.")
        annotation_box.patch.set_linewidth(0.5)
        plt.gca().add_artist(annotation_box)
        plt.show()
