import csv
import re
from collections import Counter

beginning_modifier_regex = r"^[a-z]\[\d+\]"
middle_modifier_regex = r"[A-Z]\[\d+\]"
position_regex = r"^[by](?P<position>\d+)"
charge_regex = r"\^(?P<charge>[\d])+"
neutral_loss_regex = r"[by][\d]+-(?P<neutral_loss>[\d]+)"
delta_regex = r"\/(?P<delta>-?[0-9]\d*\.*\d*)$"
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

amino_acid_modifier_replacements = {
    "C[160]": "!",
    "M[147]": "@",
    "Q[129]": "#",
    "N[115]": "$",
}

amino_acid_codes = "ACDEFGHIKLMNPQRSTVWY"
amino_acid_modifiers = "".join(amino_acid_modifier_replacements.values())
amino_acid_modified_codes = amino_acid_codes+amino_acid_modifiers


def one_hot_encode(values, code):
    result = []
    for letter in values:
        letter_encoding = [1 if code_letter == letter else 0 for code_letter in code]
        try:
            assert (sum(letter_encoding) == 1)
        except AssertionError:
            print("One hot encoding failed - unexpected letter in this name")
            print(values)
            raise
        result.append(letter_encoding)
    return result


def reverse_one_hot_encode(vectors, code):
    letters = []
    for vector in vectors:
        i = vector.index(1)  # get the index of the item which is 1
        letters.append(code[i])
    return "".join(letters)

# reverse_one_hot_encode(one_hot_encode(values, code), code) should be values.


def amino_acid_name_parse(name_str):
    """
    :param name_str: name string with beginning charge, amino acids/modifiers, and charge.
    :return: has_beginning_charge, name, charge
    """
    name, charge = name_str.split("/")
    charge = int(charge)
    beginning_modifier_search = re.search(beginning_modifier_regex, name)

    if beginning_modifier_search:
        beginning_modifier_match = beginning_modifier_search.group(0)
        if beginning_modifier_match != "n[43]":
            return False, "", 0
        else:
            has_beginning_charge = True
    else:
        has_beginning_charge = False

    if has_beginning_charge:
        name = name.replace("n[43]", "")

    for modifier, modifier_code in amino_acid_modifier_replacements.items():
        # Replace modifiers with our special codes
        name = name.replace(modifier, modifier_code)

    middle_modifier_search = re.search(middle_modifier_regex, name)
    if middle_modifier_search:
        # We found a new modifier that we haven't account for, don't.
        return False, "", 0
    name_one_hot_encoded = one_hot_encode(name, amino_acid_modified_codes)
    return has_beginning_charge, name_one_hot_encoded, charge


def reverse_amino_acid_coding(vectors, charge, has_beginning_modifier=False):
    """
    Reverse one hot encoding and character substitutions for amino acids.
    :param vectors: one hot encoded vectors
    :param has_beginning_modifier: If it should have the n[43] in front
    :return: original amino acid name string
    """
    letters = reverse_one_hot_encode(vectors, amino_acid_modified_codes)
    if has_beginning_modifier:
        letters = "n[43]"+letters
    for modifier, code in amino_acid_modifier_replacements.items():
        letters = letters.replace(code, modifier)

    letters += "/{}".format(charge)
    return letters


def create_annoying_string(row_data, row_id):
    """
    Row data is dictionary like follows:
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
    You can read it like this using CSV dictreader.
    :return: old string.
    """
    # Do some conversions
    acetyl = int(row_data['acetyl']) == 1
    name_one_hot_encoded = eval(str(row_data['name']))
    charge = int(row_data['charge'])
    precursor = float(row_data['precursor'])
    mzs = eval(str(row_data['mz']))
    intensities = eval(str(row_data['intensity']))
    ions = eval(str(row_data['ion']))
    positions = eval(str(row_data['position']))
    neutral_losses = eval(str(row_data['neutral_loss']))
    ion_charges = eval(str(row_data['ion_charge']))
    deltas = eval(str(row_data['delta']))


    # Reverse the OHE
    name = reverse_amino_acid_coding(name_one_hot_encoded, charge, acetyl)

    # Get the end string:
    ends = []
    for mz, intensity, ion, position, neutral_loss, ion_charge, delta in zip(
            mzs, intensities, ions, positions, neutral_losses, ion_charges, deltas):
        ion = indicator_codes[int(ion)]
        s = f"{mz}\t{intensity}\t{ion}{position}"
        if neutral_loss!=0:
            s += f"-{neutral_loss}"
        if ion_charge != 1:
            s += f"^{ion_charge}"
        s += f"/{delta}"
        ends.append(s)

    return f"""Name: {name}
LibID: {row_id}
MW: {round(charge*precursor, 4)}
PrecursorMZ: {precursor}
Status: Normal
FullName: {name}
Comment: No Comment
NumPeaks: {len(intensities)}\n"""+"\n".join(ends)+"\n"


def main():
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
                    print(name_str)
                    # Parse data from name.
                    has_beginning_charge, name_one_hot_encoded, charge = amino_acid_name_parse(name_str)

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

if __name__ == "__main__":
    main()
