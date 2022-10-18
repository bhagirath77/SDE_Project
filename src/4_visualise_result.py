from __future__ import print_function, division
from builtins import input
import sys
import csv
import re
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from nameparser import parse_signature
from bidi.algorithm import get_display
import questiondb
import codecs


RESULTS_DIR = 'results'
RESULTS_DIR_PNG = os.path.join(RESULTS_DIR, 'png')
RESULTS_DIR_EPS = os.path.join(RESULTS_DIR, 'eps')
RESULTS_DIR_TXT = os.path.join(RESULTS_DIR, 'txt')


def levenshtein(s, t, limit=None):
    if not s:
        return len(t)

    if not t:
        return len(s)

    if limit is not None and limit <= 0:
        return 0

    cost = 0 if s[-1] == t[-1] else 1

    return min([
        levenshtein(s[:-1], t,      limit=limit-1    if limit is not None else None) + 1,
        levenshtein(s,      t[:-1], limit=limit-1    if limit is not None else None) + 1,
        levenshtein(s[:-1], t[:-1], limit=limit-cost if limit is not None else None) + cost
    ])


class Histogram(object):
    def __init__(self, name, title, suptitle, value_transform=None, post_proc=None,
            vertical_ticks=False, horizontal_plot=False, normalize=None, ylabel='Response count',
            show_legend=True):
        self._hists = {}
        self._transform = (lambda x: x) if value_transform is None else value_transform
        self._postproc = (lambda x: None) if post_proc is None else post_proc
        self._name = name
        self._title = title
        self._suptitle = suptitle
        self._vertical_ticks = vertical_ticks
        self._horizontal_plot = horizontal_plot
        self._normalize = normalize
        self._ylabel = ylabel
        self._show_legend = show_legend

    def __len__(self, *args, **kwargs): return self._hists.__len__(*args, **kwargs)
    def __getitem__(self, *args, **kwargs): return self._hists.__getitem__(*args, **kwargs)
    def __setitem__(self, *args, **kwargs): return self._hists.__setitem__(*args, **kwargs)
    def __delitem__(self, *args, **kwargs): return self._hists.__delitem__(*args, **kwargs)
    def __contains__(self, *args, **kwargs): return self._hists.__contains__(*args, **kwargs)
    def __iter__(self, *args, **kwargs): return self._hists.__iter__(*args, **kwargs)

    def append(self, variant, value):
        hist = self._hists.get(variant, None)

        if hist is None:
            hist = {}
            self._hists[variant] = hist

        value = self._transform(value)
        if value in hist:
            hist[value] += 1  
        else:
            hist[value] = 1  

    def plot(self, texts):
        self._postproc(self)

        plot_histogram(
            self._hists, self._name, self._title, self._suptitle,
            texts, vertical_ticks=self._vertical_ticks, horizontal=self._horizontal_plot,
            normalize=self._normalize, ylabel=self._ylabel, show_legend=self._show_legend
        )


def append_histogram(histogram, value):
    if value in histogram:
        histogram[value] += 1  
    else:
        histogram[value] = 1  


def split_camel_case(name):
    matches = re.finditer(r'.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', name)
    return [m.group(0) for m in matches]


def normalize_name(name):
    camel_words = split_camel_case(name)
    final_words = []

    for camel_word in camel_words:
        w = camel_word.lower()

        w = w.replace('-', ' ').replace('.', ' ').replace('_', ' ').replace('\t', ' ')

        words = w.split()

        final_words += words

    name = '_'.join(final_words)
    return name, final_words


def normalize_function_signature(sig):
    r = parse_signature(sig)

    if not r:
        return 'invalid','invalid','invalid',['invalid']
    func_name, param_names, _ = r
    func_norm_name, func_words = normalize_name(func_name)
    param_norm_names = [normalize_name(n)[0] for n in param_names]

    return func_name, func_norm_name, func_words, param_norm_names

