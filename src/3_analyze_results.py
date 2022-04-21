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



def save_table(path, hists, title='', suptitle='', legend=None):
    keys = set()
    values = []
    for hist in hists.values():
        keys |= set(hist.keys())
        values += hist.values()

    keys = sorted(keys)
    columns = [[u''] + [str(k) for k in keys]]

    for variant in sorted(hists.keys(), reverse=True):
        hist = hists[variant]
        columns.append([variant] + [str(hist.get(key, 0)) for key in keys])
        assert(len(columns[-1]) == len(columns[0]))

    widths = [max(len(item) for item in col) for col in columns]
    sep = u'+' + u'+'.join((u'-' * (w + 2)) for w in widths) + u'+\r\n'

    def write_line(file, *fields):
        assert(len(fields) == len(widths))
        file.write(u'|' + u'|'.join((u' ' + fields[i].ljust(widths[i]) + u' ') for i in range(len(fields))) + u'|\r\n')


    with codecs.open(path, 'w', 'utf-8') as f:
        f.write(u'\ufeff') 
        f.write(u'{}\r\n{}\r\n'.format(suptitle, title))

        if legend:
            for text in legend:
                f.write(u'{}\r\n'.format(text))

        f.write(sep)

        for i in range(len(columns[0])):
            write_line(f, *[col[i] for col in columns])

            if i == 0:
                f.write(sep)

        f.write(sep)

def autolabel(ax, all_rects, font_size, horizontal=False, format='.0f'):
    """
    Attach a text label above each bar (or near its top) displaying its height
    """
    def get_value(rect):
        return rect.get_width() if horizontal else rect.get_height()

    try:
        max_height = max([max([get_value(rect) for rect in rects]) for rects, _ in all_rects])
    except:
        max_height = 1
    offset = max_height * 0.02

    format = '{:' + format + '}'

    for rects, font_color in all_rects:
        for rect in rects:
            value = get_value(rect)

            invert = value > 0 and value > 2 * offset

            pos = value + offset * (-1 if invert else 1)
            color = font_color if invert else 'k'

            if horizontal:
                y = rect.get_y() + rect.get_height() / 2.
                x = pos
                va = 'center'
                ha = 'left' if invert else 'right'
            else:
                x = rect.get_x() + rect.get_width() / 2.
                y = pos
                va = 'top' if invert else 'bottom'
                ha = 'center'

            ax.text(x,
                    y,
                    format.format(value),
                    ha=ha,
                    va=va,
                    fontdict={
                        'weight': 'bold',
                        'size': font_size,
                        'color': color
                    })


def plot_histogram(hists, name, title, suptitle, texts, vertical_ticks=False, horizontal=False,
    normalize=None, ylabel=None, show_legend=True):
    keys = set()
    values = []

    if normalize is not None:
        display_hists = {}
        for variant, hist in hists.items():
            count = sum(hist.values())
            display_hists[variant] = {k: normalize * v / count for k, v in hist.items()}
    else:
        display_hists = hists

    for hist in display_hists.values():
        keys |= set(hist.keys())
        values += hist.values()

    keys = sorted(keys, reverse=horizontal)
    x = np.arange(len(keys))

    values.append(1)
    y_max = max(values)

    dpi = 100.          
    xinch = 1920 / dpi  
    yinch = 1080 / dpi  
    fig, ax = plt.subplots(figsize=(xinch, yinch))

    spacing = 0.2
    width = (1.0 - spacing) / max(len(hists),1)
    center = (1.0 - spacing - width) / 2
    colors = ['#2874a6', '#f4d03f', '#abebc6']
    text_colors = ['w', 'k', 'k']

    legend = {}
    legend_texts = None
    all_rects = []

    i = 0

    if horizontal:
        bar = ax.barh
        set_ylim = lambda ymin, ymax: ax.set_xlim(xmin=ymin, xmax=ymax)
        set_xticks = ax.set_yticks
        set_xticklabels = ax.set_yticklabels
        set_ylabel = ax.set_xlabel
    else:
        bar = ax.bar
        set_ylim = lambda ymin, ymax: ax.set_ylim(ymin=ymin, ymax=ymax)
        set_xticks = ax.set_xticks
        set_xticklabels = lambda labels: ax.set_xticklabels(labels, rotation='vertical' if vertical_ticks else 'horizontal')
        set_ylabel = ax.set_ylabel

    all_rects = []

    for variant in sorted(display_hists.keys(), reverse=True):
        hist = display_hists[variant]
        rects = bar(
            x + i * width,
            [hist.get(key, 0) for key in keys],  
            width,
            color=colors[i % len(hists)]
        )
        legend[variant] = rects[0]  
        all_rects.append((rects, text_colors[i % len(display_hists)]))  
        i += 1

    ylim_max = y_max

    set_ylim(0, ylim_max)
    ax.set_title(title)
    set_xticks(x + center)
    set_xticklabels(get_display(str(k)) for k in keys)

    if ylabel:
        set_ylabel(ylabel)

    if horizontal:
        fig.subplots_adjust(left=0.3)
    elif vertical_ticks:
        fig.subplots_adjust(bottom=0.3)

    if show_legend:
        if texts is None:
            legend_texts_display = [u'{0} [{1}]'.format(get_display(k), sum(hists[k].values())) for k in legend.keys()]
            legend_texts = [u'{0} [{1}]'.format(k, sum(hists[k].values())) for k in legend.keys()]
        else:
            legend_texts_display = [u'{0} [{1}]: {2}'.format(get_display(k), sum(hists[k].values()), get_display(texts[k])) for k in legend.keys()]
            legend_texts = [u'{0} [{1}]: {2}'.format(k, sum(hists[k].values()), texts[k]) for k in legend.keys()]

        legend = ax.legend(
            legend.values(),
            legend_texts_display
        )

    font_size = 8 if len(ax.get_xticks()) * len(display_hists) < 50 else 5
    autolabel(ax, all_rects, font_size, horizontal=horizontal, format='.1f' if normalize is not None else '.0f')

    if not os.path.isdir(RESULTS_DIR_PNG):
        os.makedirs(RESULTS_DIR_PNG)
    if not os.path.isdir(RESULTS_DIR_EPS):
        os.makedirs(RESULTS_DIR_EPS)
    if not os.path.isdir(RESULTS_DIR_TXT):
        os.makedirs(RESULTS_DIR_TXT)

    plt.suptitle(suptitle, fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(RESULTS_DIR_PNG, name + '.png'), dpi=dpi, bbox_extra_artists=(legend,))
    plt.savefig(os.path.join(RESULTS_DIR_EPS, name + '.eps'), dpi=dpi, bbox_extra_artists=(legend,))
    save_table(os.path.join(RESULTS_DIR_TXT, name + '.txt'), display_hists, title=title, suptitle=suptitle, legend=legend_texts)

    plt.close()


class ParameterCategorizerUI(object):
    def __init__(self, question_id, question_text):
        self._dict = {}
        self._id = question_id
        self._text = question_text
        self._print_count = -1
        self.load()

        if question_id not in self._dict:
            self._dict[question_id] = {
                'cat': [],
                'map': {}
            }

    def normalize_parameters(self, sig):
        params = normalize_function_signature(sig)[3]

        normalized = []

        for p in params:
            normalized.append(self.get_category(p, sig))

        if len(normalized) == 0:
            return '(no parameters)'

        normalized.sort()
        return ', '.join(normalized)

    def _display_header(self):
        print('Question: {}'.format(self._text))
        print('Categories:', end='')

        categories = self._dict[self._id]['cat']

        for i in range(len(categories)):
            print(' [{0}] {1}'.format(i, categories[i]), end='')

        print(' [n] New [q] Quit')

    def get_category(self, name, sig):
        question_dict = self._dict[self._id]
        cat = question_dict['map'].get(name, None)

        if cat is not None:
            return cat

        if self._print_count < 0 or self._print_count >= 10:
            self._display_header()
            self._print_count = 0

        categories = question_dict['cat']
        choices = [str(i) for i in range(len(categories))] + ['n', 'q']
        response = ''

        while response not in choices:
            response = input('Category for "{0}" in "{1}": '.format(name, sig)).lower()
            self._print_count += 1

        if response == 'q':
            sys.exit()

        if response == 'n':
            cat = ''
            while not cat:
                cat = input('Category name: ')

            categories.append(cat)
            self._print_count = -1
        else:
            index = int(response)

            if index < 0 or index >= len(categories):
                raise Exception('Invalid choice')

            cat = categories[index]

        question_dict['map'][name] = cat
        self.save()

        return cat

    def save(self):
        with open('parameters.json', 'w') as f:
            json.dump(self._dict, f)

    def load(self):
        if os.path.isfile('parameters.json'):
            try:
                with open('parameters.json', 'r') as f:
                    self._dict = json.load(f)
            except IOError:
                pass


class WordReplacementUI(object):
    def __init__(self):
        self._history = {}
        self.load()  

    def should_join(self, first, second):
        r = self._history.get((first, second))

        if r is not None:
            return r

        r = self._history.get((second, first))

        if r is not None:
            return -r

        response = ''

        while response not in ['y', 'yes', 'n', 'no', '1', '2', 'q']:
            response = input('Combine "{0}" and "{1}"? [y/n/1/2/q] '.format(first, second)).lower()

        if response == 'q':
            sys.exit()

        if response in ['y', 'yes', '1', '2']:
            r = -1 if response == '2' else 1  
        else:
            r = 0  

        self._history[(first, second)] = r
        self.save()
        return r

    def save(self):
        with open('history.json', 'w') as f:
            j = [{'key': k, 'value': v} for k, v in self._history.items()]
            json.dump(j, f)

    def load(self):
        if os.path.isfile('history.json'):
            try:
                with open('history.json', 'r') as f:
                    j = json.load(f)
                    self._history = {tuple(x['key']): x['value'] for x in j}
            except IOError:
                pass


def combine_histogram_words(hist):
    ui = WordReplacementUI()

    names = list(hist.keys())[:]

    words = set()
    word_map = {}

    for name in names:
        name_words = name.split('_')
        words |= set(name_words)

    words = list(words)

    class WordInfo(object):
        def __init__(self, target):
            self.target = target

    i = 0

    while i < len(words):
        name = words[i]
        word_map[name] = WordInfo(name)

        j = i + 1

        while j < len(words):
            other_name = words[j]

            if levenshtein(name, other_name, limit=3) <= 2:
                r = ui.should_join(word_map[name].target, other_name)

                if r != 0:
                    word_map[other_name] = word_map[name]
                    del words[j]

                    if r < 0:
                        word_map[name].target = other_name

                    continue

            j += 1

        i += 1

    for name in names:
        name_words = name.split('_')
        new_words = [word_map[w].target if w in word_map else w for w in name_words]
        new_name = '_'.join(new_words)

        if new_name not in hist:
            hist[new_name] = 0

        if name != new_name:
            hist[new_name] += hist[name]
            del hist[name]


def get_question_text(text):
    text = text.replace('  ', '\n')  
    text = text.replace(u'\u2019', '\'')
    
    return text


def generate_name_histograms(rows, question_texts, question_dict, subtitle, name_prefix=''):
    if name_prefix:
        name_prefix += '_'

    def combine_words(hist):
        
        for variant in hist:
            print(variant)
            combine_histogram_words(hist[variant])

    hists = [
        Histogram(
            name_prefix + 'words', 'Number of words', subtitle,
            value_transform=lambda name: len(normalize_name(name)[1])
        ),
        Histogram(
            name_prefix + 'characters', 'Number of characters', subtitle,
            value_transform=lambda name: len(name)
        ),
        Histogram(
            name_prefix + 'names', 'Names', subtitle, vertical_ticks=True,
            value_transform=lambda name: normalize_name(name)[0],
            post_proc=combine_words
        )
    ]
    generate_histograms(rows, question_texts, question_dict, hists)


def generate_sig_histograms(rows, question_texts, question_dict, subtitle, name_prefix=''):
    if name_prefix:
        name_prefix += '_'

    def _combine_words(hist):
        for variant in hist:
            combine_histogram_words(hist[variant])

    full_id = ', '.join(sorted(question_dict.values()))
    variant = next(v for v in question_dict if 'english' in v.lower())

    if variant is None:
        raise Exception('Could not find english variant in "{0}"'.format(full_id))

    categorizer = ParameterCategorizerUI(full_id, subtitle + ': ' + \
        get_question_text(question_texts[question_dict[variant]]))

    hists = [
        Histogram(
            name_prefix + 'words', 'Number of words', subtitle,
            value_transform=lambda sig: len(normalize_function_signature(sig)[2])
        ),
        Histogram(
            name_prefix + 'characters', 'Number of characters', subtitle,
            value_transform=lambda sig: len(normalize_function_signature(sig)[0])
        ),
        Histogram(
            name_prefix + 'names', 'Names', subtitle, vertical_ticks=True,
            value_transform=lambda sig: normalize_function_signature(sig)[1],
            post_proc=_combine_words
        ),
        Histogram(
            name_prefix + 'parameters', 'Parameters', subtitle, vertical_ticks=True,
            value_transform=lambda sig: categorizer.normalize_parameters(sig)
        )
    ]
    generate_histograms(rows, question_texts, question_dict, hists)


def generate_manual_histograms(rows, question_texts, question_dict, subtitle, name_prefix=''):
    if name_prefix:
        name_prefix += '_'

    hists = [
        Histogram(
            name_prefix + 'responses', 'Responses', subtitle, horizontal_plot=True
        )
    ]
    generate_histograms(rows, question_texts, question_dict, hists)


def generate_demographic_histograms(rows, name_prefix=''):
    if name_prefix:
        name_prefix += '_'

    def determine_experience(experience):
        if experience and int(experience) >= 5:
            return 'High'
        return 'Low'


    configs = [
        {
            'questions': questiondb.get_questions(questiondb.VARNAME),
            'grouping_questions': (questiondb.EXPERIENCE),
            'hist': Histogram(
                name_prefix + 'characters_by_experience', 'Number of characters by experience', 'Demographic info',
                normalize=100,
                ylabel='Percentage',
                value_transform=lambda name: len(name)
            ),
            'texts': {
                'Low': '1st or 2nd year of Bachelor\'s degree',
                'High': '5 years or more of programming experience'
            },
            'map': determine_experience
        },
        {
            'questions': [questiondb.get_demographic_question(questiondb.EXPERIENCE)],
            'hist': Histogram(
                name_prefix + 'experience', 'Years of experience', 'Demographic info',
                value_transform=lambda experience: int(experience),
                show_legend=False
            )
        },
    ]
    basestring = str
    for config in configs:
        demographic_questions = config.get('grouping_questions', None)
        if demographic_questions is not None:
            if isinstance(demographic_questions, basestring):
                demographic_questions = (demographic_questions, )

            demographic_questions = tuple(questiondb.get_demographic_question(q) for q in demographic_questions)

        demographic_texts = config.get('texts', None)

        for row in rows:
            if demographic_questions is not None:
                value = tuple(row[q].strip() for q in demographic_questions)

                if any(value):
                    if 'map' in config:
                        value = config['map'](*value)

                        if value is None:
                            continue
                    else:
                        value = ', '.join(str(v for v in value))
                else:
                    continue

            else:
                value = ''

            for question in config['questions']:
                name = row[question].strip()

                if name:
                    config['hist'].append(value, name)

        config['hist'].plot(demographic_texts)


def generate_histograms(rows, question_texts, question_dict, hists):
    texts = {}
    x=0
    for variant in question_dict:
        try:
            texts[variant] = get_question_text(question_texts[question_dict[variant]])
        except:
            x=variant
            print(variant)
            print(question_texts)
            pass

    for row in rows:
        for variant in question_dict:
            if x==variant:
                continue
            question = question_dict[variant]
            name = row[question].strip()

            if name:  
                for hist in hists:
                    hist.append(variant, name)

    for hist in hists:
        hist.plot(texts)


def decode_row(row, encoding):
    return {k: v for k, v in row.items()}


def decode_cell_id(id):
    m = re.match('^([A-Z]+)([0-9]+)$', id.upper())

    if not m:
        raise Exception('Cannot decode cell ID "{0}"'.format(id))

    col_str = m.group(1)
    row = int(m.group(2)) - 1
    col = 0

    for i in range(len(col_str)):
        n = ord(col_str[-(i + 1)]) - ord('A') + 1
        col += n * (26 ** i)

    col -= 1

    return row, col


def main():
    global QUESTIONS_INFO

    filename = 'results-preprocessed.csv'
    updates_filename = 'results-updates.csv'
    manual_filename = 'results-manual-processed.csv'

    with open(filename, newline='') as csvfile_with_bom:
        reader = csv.DictReader(csvfile_with_bom, delimiter=',')
        headers = reader.fieldnames

        rows = []
        for row in reader:
            rows.append(row)
    with open(updates_filename, newline='') as csvfile_with_bom:
        reader = csv.DictReader(csvfile_with_bom, delimiter=',')

        for row in reader:
            row_index, col_index = decode_cell_id(row['Cell'])
            rows[row_index - 1][headers[col_index]] = row.get('Data', '')
    with open(manual_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile,delimiter=',')
        maunal_headers = reader.fieldnames

        manual_rows = []
        for row in reader:
            manual_rows.append(decode_row(row, 'utf-8'))

    if len(rows) != len(manual_rows):
        print('Row count mismatch between "{0}" and "{1}"'.format(filename, manual_filename))

    total_questions = sum(len(section.get_questions_across_variants(type)) \
        for section in questiondb.get_sections() for type in [questiondb.VARNAME, questiondb.FUNCSIG,questiondb.MANUAL])
    total_progress = 0

    print_info = {'len': 0}

    def reprint(text):
        print('\r' + ' ' * print_info['len'], end='')
        print('\r' + text, end='')

        print_info['len'] = len(text)

    def print_progress(section, type, progress, subsection_index, subsection_count):
        section_text = 'Processing section "{}": '.format(section.name)
        type_name = {
            questiondb.VARNAME: 'Variable name questions',
            questiondb.FUNCSIG: 'Function signature questions',
            questiondb.MANUAL: 'Manual classification questions'
        }[type]

        reprint('({progress:.1f}%, {qindex}/{qcount}) Processing section "{section}": {type}... ({sindex}/{scount})'.format(
            progress=100.0*float(progress+1)/float(total_questions),
            section=section.name,
            type=type_name,
            sindex=subsection_index+1,
            scount=subsection_count,
            qindex=progress+1,
            qcount=total_questions
        ))

    all_rows = []

    for i in range(len(rows)):
        row = rows[i].copy()
        row.update(manual_rows[i])
        all_rows.append(row)

    for section in questiondb.get_sections():
        section_progress = 'Processing section "{}": '.format(section.name)

        if section.has_type(questiondb.VARNAME):
            subsection_questions = section.get_questions_across_variants(questiondb.VARNAME)
            for i, questions in enumerate(subsection_questions):
                print_progress(section, questiondb.VARNAME, total_progress, i, len(subsection_questions))
                generate_name_histograms(
                    rows[1:],      
                    rows[0],       
                    questions,     
                    section.name,  
                    '{0}_var_{1}'.format(section.name.lower().replace(' ', '-'), i + 1)  
                )
                total_progress += 1

        if section.has_type(questiondb.FUNCSIG):
            
            subsection_questions = section.get_questions_across_variants(questiondb.FUNCSIG)
            for i, questions in enumerate(subsection_questions):
                
                print_progress(section, questiondb.FUNCSIG, total_progress, i, len(subsection_questions))
                generate_sig_histograms(
                    rows[1:],      
                    rows[0],       
                    questions,     
                    section.name,  
                    '{0}_sig_{1}'.format(section.name.lower().replace(' ', '-'), i + 1)  
                )
                total_progress += 1

        if section.has_type(questiondb.MANUAL):
            
            subsection_questions = section.get_questions_across_variants(questiondb.MANUAL)
            for i, questions in enumerate(subsection_questions):
                
                print_progress(section, questiondb.MANUAL, total_progress, i, len(subsection_questions))
                generate_manual_histograms(
                    manual_rows[1:], 
                    manual_rows[0],  
                    questions,       
                    section.name,    
                    '{0}_man_{1}'.format(section.name.lower().replace(' ', '-'), i + 1)  
                )
                total_progress += 1

    reprint('Done!')
    print()


if __name__ == '__main__':
    main()
