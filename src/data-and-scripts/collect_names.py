from __future__ import print_function, division, absolute_import
from asyncore import read
import sys
import csv
import re
import regex
from nameparser import parse_signature
import questiondb


def determine_suspect_name(name):
    alpha = "abcdefghijklmnopqrstuvwxyz"
    known_chars = "0123456789_" + alpha + alpha.upper()

    if len(name) > 30:
        return "Long name"

    for c in name:
        if c not in known_chars:
            return "Special character" + ((': "' + c + '"') if ord(c) < 128 else "")

    return None


def fix_signature(sig):
    if sig.startswith("("):
        sig = sig[1:] + ")"

    return sig


def match_sig(sig):
    regex.match(
        r"^(?:[^();, \t]+\s+)?(?P<func>[^();,]+)(?:\(\s*(?:(?:[^();, \t]+\s+)?(?P<param1>[^();, \t]+)\s*(?:,\s*(?:[^();, \t]+\s+)?(?P<paramn>[^();, \t]+)\s*)*)?\))?\s*;?\s*$",
        sig,
    )


def determine_suspect_signature(sig):
    alpha = "abcdefghijklmnopqrstuvwxyz"
    known_chars = "0123456789(),;_-[]*:. \t" + alpha + alpha.upper()

    if len(sig) > 100:
        return "Long name"

    for c in sig:
        if c not in known_chars:
            return "Special character" + ((': "' + c + '"') if ord(c) < 128 else "")

    if not parse_signature(sig):
        return "Pattern mismatch"

    return None


def printable_name(name):
    for c in name:
        if ord(c) >= 128:
            return "<non-printable>"

    return name


def get_cell_id(row, col):
    if col < 26:
        col = chr(ord("A") + col)
    else:
        col -= 26
        col = chr(ord("A") + col // 26) + chr(ord("A") + col % 26)

    row += 1

    return col + str(row)


def replace_html_sym(m):
    sym = m.group(1)

    return {
        "nbsp": " ",
        "lt": "<",
        "gt": ">",
        "quot": '"',
        "amp": "&",
        "apos": "'",
    }.get(sym, m.group(0))


def replace_html(v):
    return re.sub(r"&([a-z]+);", replace_html_sym, v)


class Utf8DictWriter(object):
    def __init__(self, filename=None, fieldnames=[]):
        self.filename = filename
        self.fieldnames = fieldnames
        self._file = None
        self._writer = None

    def __enter__(self):
        self.open()
        return self._writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self, filename=None):
        if self._file is not None:
            raise Exception("File already open")

        if filename is None and self.filename is None:
            raise Exception("File name not specified")

        if filename is not None:
            self.filename = filename

        self._file = open(self.filename, "wb")
        self._file.write("\ufeff".encode("utf8"))
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)

    def close(self):
        if self._writer is not None:
            self._writer = None

        if self._file is not None:
            self._file.close()
            self._file = None


def main():
    in_filename = "results-renamed.csv"
    out_filename = csv.writer(open("results-preprocessed.csv", "w"))
    updates_filename = csv.writer(open("results-updates.csv", "w"))
    other_filename = csv.writer(open("results-other.csv", "w"))

    with open(in_filename, newline="") as in_csv_bom:
        reader = csv.DictReader(in_csv_bom, delimiter=",")
        order = reader.fieldnames

        var_name_questions = questiondb.get_questions(questiondb.VARNAME)
        func_sig_questions = questiondb.get_questions(questiondb.FUNCSIG)
        all_questions = var_name_questions + func_sig_questions
        out_filename.writerow(all_questions)
        question_index = {}
        for q in all_questions:
            question_index[q] = all_questions.index(q)

        HEADER_CELL = "Cell"
        HEADER_TYPE = "Type"
        HEADER_REASON = "Reason"
        HEADER_DATA = "Data"
        UPDATES_HEADERS = [HEADER_CELL, HEADER_TYPE, HEADER_REASON, HEADER_DATA]

        updates_filename.writerow(UPDATES_HEADERS)

        other_fieldnames = [n for n in reader.fieldnames if n not in all_questions]
        other_filename.writerow(other_fieldnames)
        row_id = 0
        for row in reader:
            row_id += 1
            line = []
            line2 = []
            line3 = []
            for question in all_questions:
                val = row[question]
                type = ""
                if question in var_name_questions:
                    type = "Variable Name"
                    res = determine_suspect_name(val)
                elif question in func_sig_questions:
                    type = "Function Signature"
                    res = determine_suspect_signature(val)

                if res is not None:
                    line3.append(
                        [
                            get_cell_id(row_id, question_index[question]),
                            type,
                            res,
                            replace_html(val),
                        ]
                    )
                line.append(val)

            for other_fieldname in other_fieldnames:
                line2.append(row[other_fieldname])
            out_filename.writerow(line)
            other_filename.writerow(line2)
            for mis in line3:
                updates_filename.writerow(mis)


if __name__ == "__main__":
    main()
