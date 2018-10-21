#### subset_synth.sh
cd /home/mwu/hd/massiveKB

## pull out a subset of training and validation data
## 2M file is actually 8M (minus a few lines to have an even number of sequences)
head -n7999778 massiveKB.sptxt > massiveKB.2M.sptxt   
tail -n1999963 massiveKB.sptxt > massiveKB.2M.tail.sptxt

## need to modify script below to accept input and output filenames
python3 parse_data.py massiveKB.2M.sptxt output.csv
python3 parse_data.py massiveKB.2M.tail.sptxt output.2M.tail.csv


#### extract which equences from the greater file are synthetic or not

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



## also filter out the synthetics that were used in training data
grep "Name:" massiveKB.2M.sptxt > tmpNames.txt
grep -v "Full" tmpNames.txt > train.seq.txt

grep "Name:" massiveKB.2M.tail.sptxt > tmpNames2.txt
grep -v "Full" tmpNames2.txt > val.seq.txt

cat train.seq.txt val.seq.txt > train.val.seq.txt

sed -i 's/n\[43\]//g' train.val.seq.txt
sed -i 's/C\[160\]/C/g' train.val.seq.txt
sed -i 's/M\[147\]/M/g' train.val.seq.txt
sed -i 's/Q\[129\]/Q/g' train.val.seq.txt
sed -i 's/N\[115\]/N/g' train.val.seq.txt

sort train.val.seq.txt > tmp1
sort synth.txt > tmp2
comm -23 tmp1 tmp2 > only.unused.synth.txt




