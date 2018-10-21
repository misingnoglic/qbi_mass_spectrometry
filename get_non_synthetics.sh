#### subset_synth.sh
#### use pattern matching to pull out which of the greater file are synthetic or not

## extract the names that are not from synthetics
grep "Name:" massiveKB_noSynthetics.sptxt > nonSynthetics.txt

## extract all names in the total database
grep "Name:" massiveKB.sptxt > allNames.txt
grep -v "Full" allNames.txt > non.and.synth.txt

## remove all n\[43] C[160]  M[147] Q[129] N[115]
sed -i 's/n\[43\]//g' non.and.synth.txt
sed -i 's/C\[160\]/C/g' non.and.synth.txt
sed -i 's/M\[147\]/M/g' non.and.synth.txt
sed -i 's/Q\[129\]/Q/g' non.and.synth.txt
sed -i 's/N\[115\]/N/g' non.and.synth.txt

sort nonSynthetics.txt > non.txt
sort non.and.synth.txt > both.txt

comm -23 non.txt both.txt > synth.txt

