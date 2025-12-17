#!/usr/bin/env Rscript
# Benchmark script for original SCENT (R version)
# Runs original SCENT on benchmark datasets and measures performance

library(methods)
library(Matrix)
library(data.table)
library(parallel)
library(jsonlite)

# Load SCENT from local directory using devtools
scent_dir <- file.path(dirname(getwd()), "SCENT")
cat("Loading SCENT from:", scent_dir, "\n")

# Load SCENT package
if (!require(devtools)) {
  install.packages("devtools")
  library(devtools)
}

devtools::load_all(scent_dir, quiet = FALSE)

#' Run SCENT benchmark on a dataset
#'
#' @param data_dir Directory containing the dataset
#' @param output_dir Directory for output results
#' @param dataset_name Name of the dataset
#' @param cell_type Cell type to analyze
#' @param regression Regression type (poisson or negbin)
#' @param ncores Number of cores for parallelization
#' @param bin Whether to binarize ATAC data
#' @param seed Random seed
#'
#' @return List with benchmark results
run_benchmark <- function(
  data_dir,
  output_dir,
  dataset_name,
  cell_type = "T_cell",
  regression = "poisson",
  ncores = 6,
  bin = TRUE,
  seed = 42
) {
  cat("\n", strrep("=", 60), "\n", sep = "")
  cat("Running SCENT (R) on", dataset_name, "dataset\n")
  cat(strrep("=", 60), "\n", sep = "")

  # Set random seed
  set.seed(seed)

  # Set up paths
  dataset_path <- file.path(data_dir, dataset_name)
  rna_file <- file.path(dataset_path, "rna_matrix.csv")
  atac_file <- file.path(dataset_path, "atac_matrix.csv")
  metadata_file <- file.path(dataset_path, "metadata.csv")
  peak_info_file <- file.path(dataset_path, "peak_info.csv")

  # Check if files exist
  for (f in c(rna_file, atac_file, metadata_file, peak_info_file)) {
    if (!file.exists(f)) {
      stop(paste("Required file not found:", f))
    }
  }

  cat("Data files:\n")
  cat("  - RNA:", rna_file, "\n")
  cat("  - ATAC:", atac_file, "\n")
  cat("  - Metadata:", metadata_file, "\n")
  cat("  - Peak info:", peak_info_file, "\n")

  # Measure time for data loading
  cat("\n[1/3] Loading data...\n")
  load_start <- Sys.time()

  # Load RNA matrix
  rna_df <- fread(rna_file, header = TRUE)
  gene_names <- rna_df[[1]]
  rna_matrix <- as.matrix(rna_df[, -1])
  rownames(rna_matrix) <- gene_names
  rna_sparse <- as(rna_matrix, "dgCMatrix")

  # Load ATAC matrix
  atac_df <- fread(atac_file, header = TRUE)
  peak_names <- atac_df[[1]]
  atac_matrix <- as.matrix(atac_df[, -1])
  rownames(atac_matrix) <- peak_names
  atac_sparse <- as(atac_matrix, "dgCMatrix")

  # Load metadata
  metadata <- fread(metadata_file)

  # Load peak info
  peak_info <- fread(peak_info_file)
  colnames(peak_info) <- c("gene", "peak")
  peak_info <- as.data.frame(peak_info)

  load_time <- as.numeric(difftime(Sys.time(), load_start, units = "secs"))
  cat("  ✓ Data loaded in", round(load_time, 2), "seconds\n")

  # Get dataset statistics
  n_genes <- nrow(rna_sparse)
  n_peaks <- nrow(atac_sparse)
  n_cells <- ncol(rna_sparse)
  n_pairs <- nrow(peak_info)

  cat("\nDataset info:\n")
  cat("  - RNA matrix shape:", n_genes, "×", n_cells, "\n")
  cat("  - ATAC matrix shape:", n_peaks, "×", n_cells, "\n")
  cat("  - Gene-peak pairs:", n_pairs, "\n")

  # Create SCENT object
  cat("\nCreating SCENT object...\n")

  tryCatch({
    scent_obj <- CreateSCENTObj(
      rna = rna_sparse,
      atac = atac_sparse,
      meta.data = as.data.frame(metadata),
      peak.info = peak_info,
      covariates = c("batch", "n_counts"),
      celltypes = "cell_type"
    )
  }, error = function(e) {
    cat("Error creating SCENT object:", conditionMessage(e), "\n")
    stop(e)
  })

  cat("  ✓ SCENT object created\n")

  # Measure time for SCENT analysis
  cat("\n[2/3] Running SCENT algorithm...\n")
  cat("  - Cell type:", cell_type, "\n")
  cat("  - Regression:", regression, "\n")
  cat("  - Cores:", ncores, "\n")
  cat("  - Binarize ATAC:", bin, "\n")

  analysis_start <- Sys.time()

  tryCatch({
    scent_obj <- SCENT_algorithm(
      object = scent_obj,
      celltype = cell_type,
      ncores = ncores,
      regr = regression,
      bin = bin
    )
  }, error = function(e) {
    cat("Error running SCENT:", conditionMessage(e), "\n")
    stop(e)
  })

  analysis_time <- as.numeric(difftime(Sys.time(), analysis_start, units = "secs"))
  cat("  ✓ Analysis completed in", round(analysis_time, 2), "seconds\n")

  # Save results
  cat("\n[3/3] Saving results...\n")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  output_file <- file.path(
    output_dir,
    paste0("scent_r_", dataset_name, "_", cell_type, ".csv")
  )

  save_start <- Sys.time()

  results <- scent_obj@SCENT.result

  if (nrow(results) > 0) {
    write.csv(results, output_file, row.names = FALSE)
    save_time <- as.numeric(difftime(Sys.time(), save_start, units = "secs"))

    cat("  ✓ Results saved to:", output_file, "\n")
    cat("  ✓ Save time:", round(save_time, 2), "seconds\n")

    # Calculate statistics
    sig_001 <- sum(results$boot_basic_p < 0.01, na.rm = TRUE)
    sig_005 <- sum(results$boot_basic_p < 0.05, na.rm = TRUE)
    sig_010 <- sum(results$boot_basic_p < 0.10, na.rm = TRUE)

    cat("\nResults summary:\n")
    cat("  - Total associations tested:", nrow(results), "\n")
    cat("  - Significant at p < 0.01:", sig_001, "\n")
    cat("  - Significant at p < 0.05:", sig_005, "\n")
    cat("  - Significant at p < 0.10:", sig_010, "\n")

    # Show top 5 results
    cat("\n  Top 5 associations:\n")
    top_results <- head(results[order(results$boot_basic_p), ], 5)
    for (i in 1:min(5, nrow(top_results))) {
      row <- top_results[i, ]
      cat(sprintf("    %d. %s - %s: β=%.3f, p=%.6f\n",
                  i, row$gene, row$peak, row$beta, row$boot_basic_p))
    }
  } else {
    save_time <- 0
    sig_001 <- sig_005 <- sig_010 <- 0
    cat("  No significant results found\n")
  }

  # Calculate total time
  total_time <- load_time + analysis_time + save_time

  # Prepare benchmark statistics
  benchmark_stats <- list(
    method = "SCENT_R",
    dataset = dataset_name,
    cell_type = cell_type,
    regression = regression,
    ncores = ncores,
    bin = bin,
    n_genes = n_genes,
    n_peaks = n_peaks,
    n_cells = n_cells,
    n_pairs_tested = n_pairs,
    n_results = nrow(results),
    n_sig_001 = sig_001,
    n_sig_005 = sig_005,
    n_sig_010 = sig_010,
    load_time = load_time,
    analysis_time = analysis_time,
    save_time = save_time,
    total_time = total_time,
    seed = seed,
    r_version = paste(R.version$major, R.version$minor, sep = ".")
  )

  cat("\n", strrep("=", 60), "\n", sep = "")
  cat("Benchmark Summary\n")
  cat(strrep("=", 60), "\n", sep = "")
  cat("  Total time:", round(total_time, 2), "seconds\n")
  cat("    - Data loading:", round(load_time, 2), "s",
      sprintf("(%.1f%%)", load_time/total_time*100), "\n")
  cat("    - Analysis:", round(analysis_time, 2), "s",
      sprintf("(%.1f%%)", analysis_time/total_time*100), "\n")
  cat("    - Saving:", round(save_time, 2), "s",
      sprintf("(%.1f%%)", save_time/total_time*100), "\n")
  cat("  R version:", benchmark_stats$r_version, "\n")
  cat(strrep("=", 60), "\n\n", sep = "")

  return(benchmark_stats)
}

#' Main function
main <- function() {
  # Parse command line arguments
  args <- commandArgs(trailingOnly = TRUE)

  # Default values
  data_dir <- "data"
  output_dir <- "results/scent_r"
  datasets <- c("small")
  cell_type <- "T_cell"
  regression <- "poisson"
  ncores <- 6
  seed <- 42

  # Simple argument parsing
  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--data_dir") {
      data_dir <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--output_dir") {
      output_dir <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--datasets") {
      # Read all datasets until next argument or end
      datasets <- c()
      i <- i + 1
      while (i <= length(args) && !grepl("^--", args[i])) {
        datasets <- c(datasets, args[i])
        i <- i + 1
      }
    } else if (args[i] == "--cell_type") {
      cell_type <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--regression") {
      regression <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--ncores") {
      ncores <- as.integer(args[i + 1])
      i <- i + 2
    } else if (args[i] == "--seed") {
      seed <- as.integer(args[i + 1])
      i <- i + 2
    } else {
      i <- i + 1
    }
  }

  cat("\n", strrep("=", 60), "\n", sep = "")
  cat("SCENT (R) Benchmark\n")
  cat(strrep("=", 60), "\n", sep = "")
  cat("Configuration:\n")
  cat("  - Datasets:", paste(datasets, collapse = ", "), "\n")
  cat("  - Cell type:", cell_type, "\n")
  cat("  - Regression:", regression, "\n")
  cat("  - Cores:", ncores, "\n")
  cat("  - Random seed:", seed, "\n")
  cat("  - R version:", R.version$version.string, "\n")

  # Run benchmarks
  all_stats <- list()

  for (dataset in datasets) {
    tryCatch({
      stats <- run_benchmark(
        data_dir = data_dir,
        output_dir = output_dir,
        dataset_name = dataset,
        cell_type = cell_type,
        regression = regression,
        ncores = ncores,
        bin = TRUE,
        seed = seed
      )
      all_stats[[length(all_stats) + 1]] <- stats
    }, error = function(e) {
      cat("\nError processing", dataset, ":", conditionMessage(e), "\n")
      traceback()
    })
  }

  # Save benchmark statistics
  if (length(all_stats) > 0) {
    stats_file <- file.path(output_dir, "benchmark_stats.json")
    write_json(all_stats, stats_file, pretty = TRUE, auto_unbox = TRUE)
    cat("\n✓ Benchmark statistics saved to:", stats_file, "\n")

    # Also save as CSV
    stats_df <- do.call(rbind, lapply(all_stats, as.data.frame))
    csv_file <- file.path(output_dir, "benchmark_stats.csv")
    write.csv(stats_df, csv_file, row.names = FALSE)
    cat("✓ Benchmark statistics saved to:", csv_file, "\n")
  }

  cat("\n", strrep("=", 60), "\n", sep = "")
  cat("✓ SCENT (R) benchmark completed!\n")
  cat(strrep("=", 60), "\n\n", sep = "")
}

# Run main function
if (!interactive()) {
  main()
}
