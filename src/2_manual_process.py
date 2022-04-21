import csv

def list_answers():
    in_filename = "results-other.csv"
    with open(in_filename, newline="") as in_csv_utf16:
        reader = csv.DictReader(in_csv_utf16, delimiter=",")
        maual_headers = [
            "Q49",
            "Q50",
            "Q51",
            "Q53",
            "Q82",
            "Q83",
            "Q160",
            "Q94a",
            "Q95a",
            "Q159",
            "Q100",
            "Q101",
            "Q102",
            "Q118",
            "Q119",
            "Q157",
            "Q158",
            "Q122",
            "Q126",
        ]
        answers = {}
        for q in maual_headers:
            answers[q] = []
        row_id = 0
        reader.__next__()
        for row in reader:
            row_id += 1
            for q in maual_headers:
                if row[q] != "":
                    answers[q].append([row_id,row[q]])

        for q in maual_headers:
            print('Question:', q, len(answers[q]))
            for ans in answers[q]:
                print(ans)
            print()

list_answers()