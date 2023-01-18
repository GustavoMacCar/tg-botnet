import csv

def is_csv_empty(file):
    with open(file) as file:
        reader = csv.reader(file)
        for i, _ in enumerate(reader):
            if i:
                return False
    return True