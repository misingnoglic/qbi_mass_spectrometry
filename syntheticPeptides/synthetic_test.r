## synthetic_test.r
## Look at overlap with tryptic and non-training synthetic sequences, output for use in Michael's AI. 
## Also generate decoys in two different ways. 
## DS, 10/20/18

source("https://bioconductor.org/biocLite.R")
biocLite("MSnbase")
library('MSnbase')


## R code to compare
syn1 <- read.table("D:\\hackathon\\synthetics_seqs.txt", header=T)

syn2 <- read.table("D:\\hackathon\\only.unused.synth.txt", header=F)

syn1$seq <- gsub('",', '/', gsub(',"','', syn1$X..Charge., fixed=T), fixed=T)

hmm <- intersect(syn1$seq, syn2$V2)


syn1$charge <- substr(syn1$seq, regexpr('/', syn1$seq)+1, 99)
syn2$charge <- substr(syn2$V2, regexpr('/', syn2$V2)+1, 99)

syn1$length <- nchar(substr(syn1$seq, 1, regexpr('/', syn1$seq)-1))
syn2$length <- nchar(substr(syn2$V2, 1, regexpr('/', syn2$V2)-1))

ev <- NA
for (i in 1:5){
    evv <- read.csv(paste0("D:\\hackathon\\evidence_clean", i, ".csv"))
    ev <- rbind(ev, evv)
}
ev <- ev[-1,]

ev <- ev[!duplicated(paste0(ev$Sequence, ev$Charge)),]
ev$seq <- paste0(ev$Sequence, '/', ev$Charge)


## only use those that were not employed in training
keep <- intersect(ev$seq, syn2$V2)
write.table(keep, "D:\\hackathon\\synthetic.tryptic.peptides.txt", quote=F, row.names=F, col.names=F)

evk <- ev[ev$seq %in% keep,c('Sequence', 'seq','Length','Charge','Retention.time', 'Mass','m.z')]
evk$Sequence <- as.character(evk$Sequence)


############# scramble for decoys, recalculate mass, keep charge the same
## scramble seq, keep K and R and the end and keep composition
dec1 <- evk
dec1$pool <- substr(dec1$Sequence, 1, nchar(dec1$Sequence)-1)
dec1$last <- substr(dec1$Sequence, nchar(dec1$Sequence), 99)

set.seed(234534)
for (i in 1:nrow(dec1)){
  letters <- strsplit(dec1$pool[i], '')[[1]]
  rand <- runif(length(letters), 0, 1)
  scramble <- letters[order(rand)]
  scram <- paste0(scramble, collapse='')
  dec1$scramble[i] <- paste0(scram, dec1$last[i], '/', dec1$Charge[i])
}

  
## do same as above, change two at random (update masses)
## to preserve charge: H K R can go to each other, everything else concordant
amino_acid_codes <- "ACDEFGILMNPQSTVWY"

dec2 <- evk
dec2$pool <- substr(dec2$Sequence, 1, nchar(dec2$Sequence)-1)
dec2$last <- substr(dec2$Sequence, nchar(dec2$Sequence), 99)

set.seed(2342)
for (i in 1:nrow(dec2)){
  letters <- strsplit(dec2$pool[i], '')[[1]]
  rand <- runif(length(letters), 0, 1)
  scramble <- letters[order(rand)]
  scram <- paste0(scramble, collapse='')
  dec2$scramble[i] <- paste0(scram, dec2$last[i], '/', dec2$Charge[i])
  dec2$Mass <- calculateFragments(paste0(scram, dec2$last[i]))
  dec2$m.z <- dec2$Mass/dec2$Charge
}

evk$decoy <- 'original'
dec1$decoy <- 'decoy'
dec1$seq <- dec1$scramble
oot <- rbind(evk, dec1[,names(evk)])
names(oot)[1] <- "Original.Seq"
write.table(oot, "D:\\hackathon\\synth.peptides.plus.decoys.csv", sep=',', row.names=F, col.names=T)


## Read in Mike's MSP file of all spectra, subset to synthetic peptides.
tot <- read.table("D:\\hackathon\\FTMS-HCD-28.msp", sep='%', stringsAsFactors=F, as.is=T) 

tot$n <- 1:nrow(tot)

tot$nam <- gsub("Name: ", '', tot$V1)

tot2 <- tot[which(tot$nam %in% keep),]

keepvec <- c(tot2$n[1]:(tot2$n[1]+60))
for (i in 2:length(tot2$n)){
  keepvec <- c(keepvec, c(tot2$n[i]:(tot2$n[i]+60)))
}

tot3 <- tot[keepvec,]
tot3$seq <- NA
#tot3 <- tot3[-grep('MW:', tot3$V1),]
#tot3 <- tot3[-grep('Comment:', tot3$V1),]
#tot3 <- tot3[-grep('Num', tot3$V1),]

globvar <- NA
for (i in 1:nrow(tot3)){
  if (regexpr('Name: ', tot3[i,'V1']) > 0){
    globvar <- tot3[i,'nam']
  } 
  tot3$seq[i] <- globvar
}
tot4 <- tot3[which(tot3$seq %in% keep),]


totf <- tot[tot4$n,]
write.table(totf[,1], "D:\\hackathon\\synthetic.peptides.msp", row.names=F, col.names=F, quote=F)


