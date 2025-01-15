remotes::install_github(
  "cccnrc/plot-VCF",
  repos = BiocManager::repositories()
)

library(tidyr)
library(data.table)
library(vcfR)
library(plotVCF)
library(GenomicFeatures)
library(Biostrings)
library(ggbio)

genomestrings <- readDNAStringSet("/storage1/fs1/bga/Active/gmsroot/gc2560/core/model_data/2887491634/build21f22873ebe0486c8e6f69c15435aa96/all_sequences.fa")

gbm_bcf <- read.vcfR("GBM.combined.bcf")
gbm_vcf <- vcfR2tidy(gbm_bcf, single_frame = TRUE, info_types = TRUE, format_types = TRUE)$dat


gbm_vcf_ <- gbm_vcf %>% distinct(CHROM, POS, ID, REF, ALT, QUAL, FILTER, AF, AQ, AC, AN)
gbm_vcf_ <- gbm_vcf_ %>%
  mutate(
    ID = sapply(str_split(ID, ";"), `[`, 1),
    ALT = sapply(str_split(ALT, ","), `[`, 1),
    AF = sapply(str_split(AF, ","), `[`, 1),
    AQ = sapply(str_split(AQ, ";"), `[`, 1)
  ) %>%
  mutate(AF = as.numeric(AF))

pdf("ManhattanPlotsWES.pdf")
for (chrom in c(seq(1, 22), "X", "Y")) {
  print(chrom)
  chr_lengths <- width(genomestrings[paste0("chr", chrom)])
  seqinfo <- Seqinfo(seqnames = paste0("chr", chrom), seqlengths = chr_lengths, genome = "hg38")
  test_gr <- makeGRangesFromDataFrame(gbm_vcf_ %>% filter(CHROM == paste0("chr", chrom)), keep.extra.columns = TRUE,
                                      ignore.strand = TRUE, seqinfo = seqinfo,
                                      seqnames.field = "CHROM", start.field = "POS",
                                      end.field = "POS")
  print(plotGrandLinear(test_gr, aes(y = AF), xlab = paste0("Chromosome ", chrom), ylab = "Allelic Fraction", color = "coral") + theme_bw() + theme(legend.position = "none"))
  #plotGrandLinear(test_gr, aes(y = AF), xlab = paste0("Chromosome ", chrom), ylab = "Allelic Fraction", color = "coral") + theme_bw() + theme(legend.position = "none")

  #ggsave(paste0("/storage1/fs1/yeli/Active/chani/Data/Leuthardt_WGS_GBM_gVCFs/gVCFs/ManhattanPlots/chr", chrom, ".png"), plot = plot_to_save, width = 10, height = 8, dpi = 300)
}
dev.off()
