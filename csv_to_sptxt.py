from parse_data import create_annoying_string

import csv
def main():
    with open("predictions.csv", "r") as csv_file, open("predictions.sptxt", "w") as sptxt_file:
        csv_file = csv.DictReader(csv_file)
        row_id = 0
        for line in csv_file:
            print(create_annoying_string(line, row_id), file=sptxt_file)
            row_id+=1

if __name__ == "__main__":
    main()