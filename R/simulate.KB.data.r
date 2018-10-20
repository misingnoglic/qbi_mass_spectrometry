############ simulate.KB.data.r
## Simulate data for Michael for the training input/output.
##
## Input:
## pre_mod - indicator for n[43] up front
## name - one hot encoding for 25 letters (25 rows), across 50 positions (50 cols)
## charge - integer 1-6
## precursor mz: 300-2000 real
##
## Output:
## column of mz: 300-2000 real
## intensity: 0-1 scale
## ion: indicator for two letters
## position: integer for number, 1-L (30)
## chem_loss: one hot for 17, 44, 43, etc (say 5)
## charge: integer 1-6
## delta: -1 to 1

datDir <- "D:\\hackathon\\"

## functions to return a string, a list of lists [[0,1,0,...,0],[1,0,0,...,0]]  # each list 25 long
## one-hot encoded variable of length L
oh_l <- function(L){
  tmp <- rep(0, L)
  tmp[sample(c(1:L), 1)] <- 1
  tmp
}

## return a string that is a list of lists
## num_aa is the number of possible amino acids (plus other residues)
## seq_length is the length of the peptide sequence
seq_string <- function(num_aa, seq_length){
  st <- '['
  for (i in 1:seq_length){
    oh <- oh_l(num_aa)
    oh_string <- paste(oh, collapse=',')
    st <- paste0(st, '[', oh_string, ']')
    if (i < seq_length) { st <- paste0(st, ',')}
  }
  st <- paste0(st, ']')
  st
}

## function to turn a vector into a list string
list_string <- function(vec){
  st <- paste(vec, collapse=',')
  st <- paste0('[', st, ']')
  st
}


## set up simulated training input for 100 peptide sequences
set.seed(29387429)

sim.in <- data.frame(seq_id=1:100,
                     acetyl=sample(c(0,1), 100, replace=T),
                     name=seq_string(25, 50),
                     charge=sample(c(1:6), 100, replace=T),
                     precursor=runif(100, 300, 2000))

sim.out <- NA

## simulating 300 mz values per sequence
for (i in 1:100){
  tmp.out <- data.frame(mz=list_string(round(runif(300, 300, 2000),digits=3)),
                      intensity=list_string(round(runif(300, 0, 1),digits=3)),
                      ion=list_string(sample(c(0,1), 300, replace=T)),
                      position=list_string(sample(c(1:49), 300, replace=T)),
                      neutral_loss=list_string(sample(c(0:3), 300, replace=T)),
                      charge=list_string(sample(c(1:6), 300, replace=T)),
                      delta=list_string(round(runif(300, -.99, .99),digits=3)))
  #tmp.out <- tmp.out[order(tmp.out$mz),]
  sim.out <- rbind(sim.out, tmp.out)
}
sim.out <- sim.out[-1,]

#write.table(sim.in[which(sim.in$seq_id <= 100),], paste0(datDir, "sim.input.csv"), sep=',', row.names=F, col.names=T)
#write.table(sim.out[which(sim.out$seq_id <= 100),], paste0(datDir, "sim.output.csv"), sep=',', row.names=F, col.names=T)

sim.data <- cbind(sim.in, sim.out)
write.table(sim.data, paste0(datDir, "sim.data.csv"), sep=',', row.names=F, col.names=T)

