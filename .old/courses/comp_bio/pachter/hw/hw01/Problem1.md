# Problem 1 Responses
1. UMIs are used to identify unique molecules (transcripts) to avoid duplicated counts of the same original transcript.
2. A "library" is the prepared set of cDNA, where each transcript is an element in the library.
3. inDrops and 10X use "sub-Poisson" (aka "super-Poisson", not confusing at all [[error discussion](https://liorpachter.wordpress.com/2019/02/07/sub-poisson-loading-for-single-cell-rna-seq/)]) loading techniques.
The technique is dubbed as such because the variance in beads loaded to each droplet is reduced in comparison to the traditional plastic bead technique ([ref](https://liorpachter.wordpress.com/2019/02/07/sub-poisson-loading-for-single-cell-rna-seq/)).
4. 3' capture is a reference to the end of the polyadenylated transcript that is captured in the library.
5. Order of events in 10x:
    1. Cell capture and lysis
    2. 3’ transcript capture and barcoding
    3. Reverse transcription and amplification
    4. cDNA fragmentation and size selection
    5. Addition of sample index/label (sample index PCR)
    6. Single- or paired-end sequencing
