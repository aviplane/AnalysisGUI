from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from FormattingStrings import *
from plotformatting import *
import numpy as np
import AnalysisFunctions as af
import traceback
import math
from colorcet import cm
from sklearn.decomposition import PCA
from scipy.stats import norm


class PlotFitWorker(QRunnable):
    def __init__(self, current_folder, fig, xlabel, units, fit_mean, fit_std, roi_labels, keys_adjusted, rois_to_exclude=[]):
        """

        """
        super(PlotFitWorker, self).__init__()
        self.current_folder = current_folder
        self.fig = fig
        self.xlabel = xlabel
        self.units = units
        self.fit_mean = fit_mean
        self.fit_std = fit_std
        self.roi_labels = roi_labels
        self.keys_adjusted = keys_adjusted
        self.rois_to_exclude = rois_to_exclude


class Plot2DWorker(PlotFitWorker):
    def run(self):
        try:
            self.fig.clf()
            rois_to_plot = [i for i in range(
                len(self.roi_labels)) if self.roi_labels[i] not in self.rois_to_exclude]
            n_rows, n_columns = 2, math.ceil(len(rois_to_plot) / 2)
            if len(self.keys_adjusted) < 2:
                return
            extent = [-0.5, self.fit_mean.shape[2] + 0.5,
                      np.max(self.keys_adjusted) +
                      np.diff(self.keys_adjusted)[0],
                      np.min(self.keys_adjusted)]
            for e, i in enumerate(rois_to_plot):
                label = self.roi_labels[i]
                ax = self.fig.add_subplot(n_rows, n_columns, e + 1)
                cax = ax.imshow(self.fit_mean[i], aspect="auto",
                                cmap=cm.blues, extent=extent, vmin=0)
                af.save_array(self.fit_mean[i], label, self.current_folder)
                self.fig.colorbar(cax, ax=ax, label="Fitted counts")
                if label in fancy_titles.keys():
                    ax.set_title(fancy_titles[label])
                else:
                    ax.set_title(label)
                ax.set_xlabel("Trap Index")
                ax.set_ylabel(f"{self.xlabel} ({self.units})")
            af.save_figure(self.fig, "2d_plot", self.current_folder)
        except:
            traceback.print_exc()


class Plot1DWorker(PlotFitWorker):
    """
    Worker Thread to make 1D plot
    """

    def run(self):
        try:
            self.fig.clf()
            axis = self.fig.add_subplot(111)
            for state, state_std, label in zip(self.fit_mean, self.fit_std, self.roi_labels):
                if label not in self.rois_to_exclude:
                    transparent_edge_plot(axis,
                                          self.keys_adjusted,
                                          np.mean(state, axis=1),
                                          np.mean(state_std, axis=1),
                                          label=fancy_titles[label])
                    af.save_array(np.mean(state, axis=1),
                                  f"{label}_1d", self.current_folder)
                    af.save_array(np.mean(state, axis=1),
                                  f"{label}_1d_std", self.current_folder)
            axis.axhline(self.f2_threshold, color='r',
                         linestyle='--', label="F = 2 Threshold")
            axis.legend()
            axis.set_ylabel("Average trap counts")
            axis.set_xlabel(f"{self.xlabel} ({self.units})")

            af.save_figure(self.fig, "1d_plot", self.current_folder)

        except:
            traceback.print_exc()


class Plot1DHistogramWorker(PlotFitWorker):
    def run(self):
        try:
            colors = ["tab:blue", "tab:orange", "tab:green",
                      "tab:red", "tab:purple", "tab:brown", "tab:pink"]
            self.fig.clf()
            axis = self.fig.add_subplot(111)
            for state, state_std, label, color in zip(self.fit_mean, self.fit_std, self.roi_labels, colors):
                if label not in self.rois_to_exclude:
                    data = np.mean(state, axis=1)
                    mean, std = norm.fit(data)
                    out, bins, patches = axis.hist(data, bins='stone',
                                                   label=f"{fancy_titles[label]} - Mean: {mean:.0f}, Std.: {std:.1f}",
                                                   color=color)
                    x = np.linspace(np.min(data), np.max(data))
                    axis.plot(x, np.max(out) * norm.pdf(x, mean, std), c=color)

            axis.legend()
            axis.set_ylabel("Number of Shots")
            axis.set_xlabel(f"Imaging Counts")
            af.save_figure(self.fig, "1d_histplot", self.current_folder)
        except:
            traceback.print_exc()


class PlotPCAWorker(PlotFitWorker):
    def run(self):
        print(self.fit_mean.shape)
        if self.fit_mean.shape[0] < 8:
            return
        n_traps = self.fit_mean.shape[2]
        self.roi_labels = list(self.roi_labels)
        fit_1m1 = self.fit_mean[:, self.roi_labels.index('roi1-1')]  # fit_sum
        fit_1p1 = self.fit_mean[:, self.roi_labels.index('roi11')]  # fit_sum
        fit_10 = self.fit_mean[:, self.roi_labels.index('roi10')]
        fit_sum = (fit_1m1 + fit_1p1 + fit_10)
        fit_1m1, fit_1p1 = fit_1m1 / fit_sum, fit_1p1 / fit_sum

        fit_side_mode = np.hstack([fit_1m1, fit_1p1])
        n_components = 5
        side_mode_pca = PCA(n_components=n_components)
        side_mode_pca.fit(fit_side_mode)

        variance_explanation = side_mode_pca.explained_variance_ratio_
        components = side_mode_pca.components_

        self.fig.clf()
        ax_num = 1
        for component, variance in zip(components, variance_explanation):
            ax = self.fig.add_subplot(n_components, 1, ax_num)
            c = [component[:n_traps], component[n_traps:]]
            mag = np.max(np.abs(c))
            cax = ax.imshow(c, cmap=cm.coolwarm, vmin=-mag, vmax=mag)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["1, -1", "1, 1"])
            ax.set_title(f"Variance explained: {variance * 100:.1f}%")
            self.fig.colorbar(cax, ax=ax)
            ax.set_xlabel("Trap Index")
            ax_num += 1

        af.save_figure(self.fig, "pc_pca_copmonents", self.current_folder)


class PlotCorrelationWorker(PlotFitWorker):
    def set_limits(self, correlation_low, correlation_high):
        self.threshold_low = correlation_low
        self.threshold_high = correlation_high

    def run(self):
        if self.fit_mean.shape[0] < 2:
            return
        self.roi_labels = list(self.roi_labels)
        fit_10 = self.fit_mean[:, self.roi_labels.index('roi10')]
        fit_1m1 = self.fit_mean[:, self.roi_labels.index('roi1-1')]
        fit_1p1 = self.fit_mean[:, self.roi_labels.index('roi11')]
        n_traps = fit_10.shape[1]
        sidemode = np.mean(
            (fit_1m1 + fit_1p1) / (fit_10 + fit_1m1 + fit_1p1),
            axis=1
        )

        print(sidemode, sidemode.shape)
        threshold_index = np.where((sidemode >= self.threshold_low) &
                                   (sidemode < self.threshold_high))

        if self.threshold_low > self.threshold_high:
            return
        fit_1m1 = fit_1m1[threshold_index]
        fit_1p1 = fit_1p1[threshold_index]
        fit_10 = fit_10[threshold_index]
        fit_sum = fit_10 + fit_1m1 + fit_1p1
        fit_sidemode = fit_1m1 + fit_1p1
        corr = np.corrcoef(
            (fit_1m1 / fit_sum).T,
            (fit_1p1 / fit_sum).T)[:n_traps, n_traps:]
        corr_sidemode = np.corrcoef(
            (fit_1m1 / fit_sidemode).T,
            (fit_1p1 / fit_sidemode).T)[:n_traps, n_traps:]
        self.fig.clf()
        axes = (self.fig.add_subplot(2, 2, 1),
                self.fig.add_subplot(2, 2, 2),
                self.fig.add_subplot(2, 2, 3),
                self.fig.add_subplot(2, 2, 4),
                )
        cax = axes[0].imshow(
            corr,
            aspect="equal",
            interpolation="None",
            vmin=-1,
            vmax=1,
            cmap=correlation_colormap)
        axes[0].set_xlabel("1, -1 trap index")
        axes[0].set_ylabel("1, 1 trap index")
        axes[0].set_title("Total Atom Number Normalization")
        self.fig.colorbar(cax, ax=axes[0])
        af.save_array(corr, "total_norm_corr", self.current_folder)
        cax = axes[1].imshow(
            corr_sidemode,
            aspect="equal",
            interpolation="None",
            vmin=-1,
            vmax=1,
            cmap=correlation_colormap)
        axes[1].set_xlabel("1, -1 trap index")
        axes[1].set_ylabel("1, 1 trap index")
        axes[1].set_title("Sidemode Normalization")
        self.fig.colorbar(cax, ax=axes[1])
        af.save_array(corr_sidemode, "sidemode_norm_corr", self.current_folder)
        positions = list(range(-n_traps + 1, n_traps))
        total_diag = [np.mean(np.diagonal(corr, d)) for d in positions]
        sidemode_diag = [np.mean(np.diagonal(corr_sidemode, d))
                         for d in positions]
        axes_num = 2
        for diag in [total_diag, sidemode_diag]:
            axes[axes_num].plot(positions, diag, 'o--')
            axes[axes_num].set_xlabel("Distance (sites)")
            axes[axes_num].set_ylabel("Correlation")
            axes[axes_num].set_ylim(-1, 1)
            axes_num += 1
        af.save_array(total_diag, "total_norm_corr_1d", self.current_folder)
        af.save_array(sidemode_diag, "sidemode_norm_corr_1d",
                      self.current_folder)
        axes[2].set_title("Total atom number normalization")
        axes[3].set_title("Sidemode normalization")
        af.save_figure(self.fig, "correlations", self.current_folder)
