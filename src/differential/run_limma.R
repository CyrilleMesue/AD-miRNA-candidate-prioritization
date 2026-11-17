#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

expr_file <- args[1]          # Path to expression matrix
meta_file <- args[2]          # Path to sample metadata
group_col <- args[3]          # Column name in metadata for group assignment
output_file <- args[4]        # Output file path
reference_group <- ifelse(length(args) >= 5, args[5], "Control")  # Optional 5th arg

library(limma)
library(edgeR)

# Read input files
expr <- read.csv(expr_file, row.names = 1, check.names = FALSE)
meta <- read.csv(meta_file, row.names = 1)

# Match sample order
meta <- meta[colnames(expr), , drop = FALSE]

# Set group factor and relevel
group <- as.factor(meta[[group_col]])
group <- relevel(group, ref = reference_group)

# Design matrix with no intercept
design <- model.matrix(~0 + group)
colnames(design) <- levels(group)

library(limma)
library(edgeR)

# read expr, meta, build 'group' and 'design' as you already do...

# Heuristic: are these raw counts?
is_integer_like <- function(x) {
  all(is.finite(x)) && all(abs(x - round(x)) < 1e-8) && min(x) >= 0
}

X <- as.matrix(expr)

if (is_integer_like(X)) {
  # -------- RAW COUNTS: use voom ----------
  y <- DGEList(counts = X)
  keep <- filterByExpr(y, design)     # optional but recommended
  y <- y[keep, , keep.lib.sizes = FALSE]
  y <- calcNormFactors(y, method = "TMM")
  v <- voom(y, design)                # mean-variance modeling from counts
  fit <- lmFit(v, design)
  use_trend <- FALSE
} else {
  # -------- ALREADY NORMALIZED / LOG VALUES: skip voom ----------
  # Expect log2-scale (e.g., log2(TPM+offset) or logCPM).
  # Fit with limma-trend to account for mean-variance trend.
  fit <- lmFit(X, design)
  use_trend <- TRUE
}

# contrasts (your existing branch for 2 groups)
if (ncol(design) == 2) {
  other_group <- setdiff(colnames(design), reference_group)
  contrast <- makeContrasts(contrasts = paste(other_group, "-", reference_group),
                            levels = design)
  fit2 <- contrasts.fit(fit, contrast)
} else {
  fit2 <- fit
}

fit2 <- eBayes(fit2, trend = use_trend)  # trend=TRUE only for non-voom path
res <- topTable(fit2, number = Inf, adjust.method = "BH")
write.csv(res, file = output_file, row.names = TRUE)

