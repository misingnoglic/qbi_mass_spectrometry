## calculate_frag_mz.py
## collection of functions to calculate fragment m/z from the peptide sequence and ion information

from pyteomics import mass

amino_acid_modifier_replacements = {
    "C[160]": "!",
    "M[147]": "@",
    "Q[129]": "#",
    "N[115]": "$",
}

dumb_reversal = {
    '!': 'B',
    '@': 'J',
    '#': 'O',
    '$': 'U',
}

amino_acid_codes = "ACDEFGHIKLMNPQRSTVWY"
amino_acid_modifiers = "".join(amino_acid_modifier_replacements.values())
amino_acid_modified_codes = amino_acid_codes+amino_acid_modifiers

## set new dictionary elements for the characters we defined for modified amino acids
aa_comp = dict(mass.std_aa_comp)

## go from our new coding back to chemical formula
#   "C[160]": "!",   cystine + Carbamidomethyl
#   "M[147]": "@",   methionine + oxidation
#   "Q[129]": "#",   glutamine + deamidation
#   "N[115]": "$",   asparagine + deamidation

aa_comp["B"] = mass.Composition('C5H8N2O2S1')  ## these need chemical formulas for C[160], etc
aa_comp["J"] = mass.Composition('C5H9N1O2S1')
aa_comp["O"] = mass.Composition('C5H7N1O3')
aa_comp["U"] = mass.Composition('C4H5N1O3')


def reverse_one_hot_encode(vectors, code):
    letters = []
    for vector in vectors:
        i = vector.index(1)  # get the index of the item which is 1
        letters.append(code[i])
    return "".join(letters)
  
def get_frag_mz(one_hot, ion_position, ion_type, ion_charge):
  pep_seq = reverse_one_hot_encode(one_hot, amino_acid_modified_codes)
  
  if ion_type == 'b':
    ion_seq = pep_seq[:ion_position]
  elif ion_type == 'y':
    ion_seq = pep_seq[-ion_position:]

  # # TODO: Figure out how to use aacomp
  # count = ion_seq.count("!")
  # ion_seq =ion_seq.replace("!", "C")
  for key, value in dumb_reversal.items():
      ion_seq = ion_seq.replace(key, value)
  mz = mass.calculate_mass(sequence=ion_seq, ion_type=ion_type, charge=int(ion_charge), aa_comp=aa_comp)
  # mz += count * float(2*12 + 1 + 14)/int(ion_charge)
  return mz


  
