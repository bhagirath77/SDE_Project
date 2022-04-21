from __future__ import print_function, division
import sys
import csv

RENAME_RULES = {
    'Q81': ['Q81a', 'Q81b'],
    'Q94': ['Q94a', 'Q94b'],
    'Q95': ['Q95a', 'Q95b'],
    'Q96': ['Q96a', 'Q96b']
}


def rename_header(row):
    for i in range(len(row)):
        v = row[i].strip()
        replacements = RENAME_RULES.get(v, None)

        if replacements is not None:
            if len(replacements) == 0:
                raise Exception('Not enough replacements for {0}'.format(v))

            v = replacements[0]
            del replacements[0]

        row[i] = v


def main():
    global VAR_NAME_QUESTIONS

    in_filename = 'results.csv'
    out_filename = 'results-renamed.csv'

    with open(in_filename, newline='') as in_csv_utf16:
        reader = csv.reader(in_csv_utf16, delimiter=',')

        with open(out_filename, 'w') as out_csv:
            out_csv.write(u'\ufeff')
            writer = csv.writer(out_csv)

            is_header = True
            for row in reader:
                if is_header:
                    rename_header(row)
                    is_header = False
                writer.writerow(row)


if __name__ == '__main__':
    main()
