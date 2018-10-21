#source("https://bioconductor.org/biocLite.R")
#biocLite("MSnbase")
library(MSnbase)
calculateFragments('SAMPLEPEPTIDQE',z=c(1,2), verbose=T, modifications=c(Nterm=43))
defaultNeutralLoss()

wd = "/Users/mtrnka/Dropbox/Conferences/Hackathon/syntheticPeptides/"
setwd(wd)
folders <- list.files(patter="TUM.*")
synthetics <- data.frame(Sequence=character(),Modified.sequence=character(),Charge=integer(),fileName=character())
for (fo in folders) {
  setwd(fo)
  ev <- read.csv("evidence_clean.csv")
#  print(summary(ev$Modifications))
  ev2 <- ev[,c("Sequence","Modified.sequence","Charge")]
  ev2$Modified.sequence <- gsub("_","",ev2$Modified.sequence)
  ev2$fileName <- fo
  synthetics <- rbind(synthetics,ev2)
  #  peps <- levels(ev$Modified.sequence)
#  peps <- gsub("_","",peps)
#  filName = fo
#  filName = gsub("3xHCD-1h-R1-tryptic","uniquePeps.txt",filName)
#  nop = paste("numberPeptides=",length(peps),sep="")
#  cat(nop,file=filName,sep="\n")
#  cat(peps,file=filName,sep="\n",append=T)
  setwd(wd)
}

setwd("/Users/mtrnka/Dropbox/Conferences/Hackathon/qbi_mass_spectrometry/")
unUsed <- read.table("only.unused.synth.txt",sep=" ")
unUsed$V2 <- as.character(unUsed$V2)
unUsed <- strsplit(unUsed$V2,split = '/')
head(unUsed)
unUsedA <- sapply(unUsed,function(x){x[1]})
unUsedB <- sapply(unUsed,function(x){x[2]})
unUsed <- as.data.frame(cbind(unUsedA,unUsedB))
names(unUsed) <- c("Sequence","Charge")
unUsed$Used <- T
synthetics2 <- merge(synthetics, unUsed, all.x=T)
synthetics.used <- synthetics2[which(synthetics2$Used),]
length(unique(synthetics.used$Sequence))
