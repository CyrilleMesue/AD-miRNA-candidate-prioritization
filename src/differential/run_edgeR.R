#!/usr/bin/env Rscript

# Usage:
# Rscript edgeR_standard.R counts.csv meta.csv group_col output.csv reference_group
#
# Example:
# Rscript edgeR_standard.R counts.csv samples.csv group results_edgeR.csv CTRL

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("Usage: Rscript edgeR_standard.R counts.csv meta.csv group_col output.csv reference_group", call. = FALSE)
}

counts_file     <- args[1] # Path to expression matrix. Must be raw counts
meta_file       <- args[2] # Path to sample metadata
group_col       <- args[3] # Column name in metadata for group assignment
output_file     <- args[4] # Output file path
reference_group <- ifelse(length(args) >= 5, args[5], "Control")  # Optional 5th arg

suppressPackageStartupMessages({
  library(edgeR)
})

# ---- Read input ----
expr <- read.csv(counts_file, row.names = 1, check.names = FALSE)
meta <- read.csv(meta_file,  row.names = 1, check.names = FALSE)

# Match sample order
if (!all(colnames(expr) %in% rownames(meta))) {
  stop("Not all expression columns are present in metadata rownames.", call. = FALSE)
}
meta <- meta[colnames(expr), , drop = FALSE]

# Group factor and reference
if (!(group_col %in% colnames(meta))) {
  stop(sprintf("Column '%s' not found in metadata.", group_col), call. = FALSE)
}
group <- factor(meta[[group_col]])
if (!(reference_group %in% levels(group))) {
  stop(sprintf("Reference group '%s' not found in metadata column '%s'.",
               reference_group, group_col), call. = FALSE)
}
group <- relevel(group, ref = reference_group)

# ---- edgeR standard workflow ----
y <- DGEList(counts = as.matrix(expr), group = group)

keep <- filterByExpr(y, group = group)
y <- y[keep, , keep.lib.sizes = FALSE]

y <- calcNormFactors(y, method = "TMM")

design <- model.matrix(~ group)

y <- estimateDisp(y, design)

fit <- glmQLFit(y, design, robust = TRUE)

# Test: non-reference vs reference (assumes 2 groups)
coef_to_test <- which(!colnames(design) %in% "(Intercept)" & colnames(design) != paste0("group", reference_group))
if (length(coef_to_test) != 1) {
  stop("Design must be two-group (reference vs one other group).", call. = FALSE)
}

qlf <- glmQLFTest(fit, coef = coef_to_test)

res <- topTags(qlf, n = Inf)$table  # logFC = (non-ref vs ref)

write.csv(res, file = output_file, row.names = TRUE)
