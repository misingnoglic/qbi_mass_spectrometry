import csv
import re
from collections import Counter

beginning_modifier_regex = r"^[a-z]\[\d+\]"
middle_modifier_regex = r"[A-Z]\[\d+\]"
position_regex = r"^[by](?P<position>\d+)"
charge_regex = r"\^(?P<charge>[\d])+"
neutral_loss_regex = r"[by][\d]+-(?P<neutral_loss>[\d]+)"
delta_regex = r"\/(?P<delta>-?[0-9]\d*\.*\d*)$"
amino_acid_codes = "ACDEFGHIKLMNPQRSTVWY"
amino_acid_modifiers = "!@#$"
amino_acid_modified_codes = amino_acid_codes+amino_acid_modifiers
indicator_codes = "by"
csv_output_rows = [
    'acetyl',
    'name',
    'charge',
    'precursor',
    'mz',
    'intensity',
    'ion',
    'position',
    'neutral_loss',
    'ion_charge',
    'delta',
]


def one_hot_encode(values, code):
    result = []
    for letter in values:
        letter_encoding = [1 if code_letter == letter else 0 for code_letter in code]
        try:
            assert (sum(letter_encoding) == 1)
        except AssertionError:
            print("One hot encoding failed - unexpected letter in this name")
            print(name)
            raise
        result.append(letter_encoding)
    return result


with open("example.KB.sptxt") as spec_data_file, open("output.csv", "w", newline='') as csv_output_file:
    writer = csv.DictWriter(csv_output_file, fieldnames=csv_output_rows)
    writer.writeheader()
    while True:
        current_chunk = []
        read = False
        for line in spec_data_file:
            read = True
            if not line.strip():
                # End of chunk - start processing

                # Variables to write:
                # has_beginning_modifier
                # encoded_name
                # charge

                status = current_chunk[4].split()[1]
                if status!="Normal":
                    print("NOT NORMAL")
                    break

                name_str = current_chunk[0].split(" ")[1]
                name, charge = name_str.split("/")
                charge = int(charge)
                beginning_modifier_search = re.search(beginning_modifier_regex, name)

                if beginning_modifier_search:
                    beginning_modifier_match = beginning_modifier_search.group(0)
                    if beginning_modifier_match != "n[43]":
                        break # Don't include this sample
                    else:
                        has_beginning_charge = True
                else:
                    has_beginning_charge = False

                if has_beginning_charge:
                    name = name.replace("n[43]", "")

                name = name.replace("C[160]", "!")
                name = name.replace("M[147]", "@")
                name = name.replace("Q[129]", "#")
                name = name.replace("N[115]", "$")
                middle_modifier_search = re.search(middle_modifier_regex, name)
                if middle_modifier_search:
                    # We found a new modifier that we haven't account for, don't.
                    break
                # middle_modifiers = re.findall(middle_modifier_regex, name)
                # middle_modifiers_all.extend(middle_modifiers)
                name_one_hot_encoded = one_hot_encode(name, amino_acid_modified_codes)

                precursor_mz_line = current_chunk[3]
                try:
                    assert(precursor_mz_line.startswith("PrecursorMZ: "))
                except AssertionError:
                    print("This chunk has some extra lines... or some deleted lines")
                    print(current_chunk)
                    break

                precursor_mz = float(precursor_mz_line.split()[1])

                m_over_z = []
                intensities = []
                ions = []
                ion_charges = []
                neutral_losses = []
                deltas = []
                positions = []

                for data in current_chunk[8:]:
                    mz, intensity, ion_data = data.split()
                    ion_data = ion_data.split(",")[0]  # Only care about first item
                    if ion_data[0] not in indicator_codes:
                        continue  # Skip this row
                    if "i" in ion_data:
                        continue
                    ion_indicator_code = indicator_codes.index(ion_data[0])
                    position_search = re.search(position_regex, ion_data)
                    position = int(position_search.group("position"))
                    positions.append(position)
                    charge_search = re.search(charge_regex, ion_data)
                    if charge_search:
                        ion_charge = int(charge_search.group("charge"))
                    else:
                        ion_charge = 1

                    neutral_loss_search = re.search(neutral_loss_regex, ion_data)
                    if neutral_loss_search:
                        neutral_loss = int(neutral_loss_search.group("neutral_loss"))
                    else:
                        neutral_loss = 0

                    delta = float(ion_data.split("/")[1])
                    delta = round(delta, 3)

                    m_over_z.append(round(float(mz), 3))
                    intensities.append(round(float(intensity), 3))
                    ions.append(ion_indicator_code)
                    ion_charges.append(ion_charge)
                    neutral_losses.append(neutral_loss)
                    deltas.append(delta)

                row_dictionary = {
                    'acetyl': int(has_beginning_charge),
                    'name': name_one_hot_encoded,
                    'charge': charge,
                    'precursor': precursor_mz,
                    'mz': m_over_z,
                    'intensity': intensities,
                    'ion': ions,
                    'position': positions,
                    'neutral_loss': neutral_losses,
                    'ion_charge': ion_charges,
                    'delta': deltas,
                }
                assert len(m_over_z)==len(intensities)==len(ions)==len(neutral_losses)==len(ion_charges)==len(deltas)
                writer.writerow(row_dictionary)
                break
            else:
                current_chunk.append(line.strip())
        if not read:
            break
