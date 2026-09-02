version 1.0

# Run CellBender for every sample found in input_gcs_dir (or just the given
# `samples` list, if set), across the given modes. Expects one input file per
# (sample, mode) at
# {input_gcs_dir}/{sample}.{mode}.h5ad (mirrors code/make_h5_mdl1889.sh
# naming), and writes each sample's outputs to {output_gcs_dir}/{sample}/.
# Not tied to any one dataset: point input_gcs_dir/output_gcs_dir at whatever
# bucket paths hold that dataset's h5ad files.

workflow RunCellbender {
  input {
    String input_gcs_dir
    String output_gcs_dir
    String docker_image
    Array[String]? samples
    Array[String] modes = ["gex", "all"]
    # Each of these three sweeps independently: give one of them multiple
    # values to test that parameter, and leave the other two at their
    # single-value defaults so the run count stays just the sweep size
    # (not a cross product of all three).
    Array[Int] epochs = [35]
    Int expected_cells = 100000
    Int total_droplets_included = 200000
    Array[Float] training_fraction = [0.9]
    Array[Int] seed = [1234]
    Int cpu = 16
    Int memory_gb = 64
    Int disk_gb = 100
  }

  if (!defined(samples)) {
    call DiscoverSamples { input: input_gcs_dir = input_gcs_dir, modes = modes }
  }
  Array[String] samples_to_run = select_first([samples, DiscoverSamples.samples])

  scatter (sample in samples_to_run) {
    scatter (mode in modes) {
      scatter (epochs_ in epochs) {
        scatter (seed_ in seed) {
          scatter (training_fraction_ in training_fraction) {
            call RunCellbender {
              input:
                sample        = sample,
                mode          = mode,
                input_gcs_dir = input_gcs_dir,
                docker_image  = docker_image,
                epochs        = epochs_,
                expected_cells = expected_cells,
                total_droplets_included = total_droplets_included,
                training_fraction = training_fraction_,
                seed          = seed_,
                cpu           = cpu,
                memory_gb     = memory_gb,
                disk_gb       = disk_gb
            }
          }
        }
      }
    }

    call OrganizeSample {
      input:
        sample                       = sample,
        h5_files                     = flatten(flatten(flatten(RunCellbender.output_h5))),
        filtered_h5_files            = flatten(flatten(flatten(RunCellbender.filtered_h5))),
        pdfs                         = flatten(flatten(flatten(RunCellbender.pdf))),
        cell_barcodes_csvs           = flatten(flatten(flatten(RunCellbender.cell_barcodes_csv))),
        metrics_csvs                 = flatten(flatten(flatten(RunCellbender.metrics_csv))),
        posterior_parquets           = flatten(flatten(flatten(RunCellbender.posterior_parquet))),
        posterior_latents_csvs       = flatten(flatten(flatten(RunCellbender.posterior_latents_csv))),
        posterior_global_latents_jsons = flatten(flatten(flatten(RunCellbender.posterior_global_latents_json))),
        report_htmls                 = flatten(flatten(flatten(RunCellbender.report_html))),
        logs                         = flatten(flatten(flatten(RunCellbender.log))),
        reports                      = flatten(flatten(flatten(RunCellbender.resource_report))),
        output_gcs_dir               = output_gcs_dir
    }
  }
}

task DiscoverSamples {
  input {
    String input_gcs_dir
    Array[String] modes
  }

  command <<<
    set -euo pipefail
    for MODE in ~{sep=" " modes}; do
      gsutil ls "~{input_gcs_dir}"/*."${MODE}".h5ad | while read -r PATH_; do
        basename "$PATH_" ".${MODE}.h5ad"
      done
    done | sort -u > samples.txt
  >>>

  output {
    Array[String] samples = read_lines("samples.txt")
  }

  runtime {
    docker: "google/cloud-sdk:slim"
    cpu: 1
    memory: "2 GB"
    disks: "local-disk 10 HDD"
  }
}

task RunCellbender {
  input {
    String sample
    String mode
    String input_gcs_dir
    String docker_image
    Int epochs
    Int expected_cells
    Int total_droplets_included
    Float training_fraction
    Int seed
    Int cpu
    Int memory_gb
    Int disk_gb
  }

  File input_h5ad = input_gcs_dir + "/" + sample + "." + mode + ".h5ad"
  String out_prefix = "~{sample}.~{mode}.e~{epochs}.s~{seed}.tf~{training_fraction}"

  command <<<
    set -euo pipefail
    START_TIME=$(date +%s)

    cellbender remove-background \
      --input ~{input_h5ad} \
      --output ~{out_prefix}.cellbender.h5 \
      --cpu-threads ~{cpu} \
      --epochs ~{epochs} \
      --expected-cells ~{expected_cells} \
      --total-droplets-included ~{total_droplets_included} \
      --training-fraction ~{training_fraction} \
      --random-seed ~{seed} \
      > >(tee ~{out_prefix}.log) 2>&1 &
    CB_PID=$!

    PEAK_KB=0
    while kill -0 "$CB_PID" 2>/dev/null; do
      STATUS_LINE="$(grep -m1 '^VmHWM:' "/proc/$CB_PID/status" 2>/dev/null || true)"
      CUR_KB="$(awk '{print $2}' <<< "$STATUS_LINE")"
      if [[ -n "$CUR_KB" && "$CUR_KB" -gt "$PEAK_KB" ]]; then
        PEAK_KB="$CUR_KB"
      fi
      sleep 2
    done

    CB_EXIT=0
    wait "$CB_PID" || CB_EXIT=$?
    ELAPSED_SEC=$(( $(date +%s) - START_TIME ))

    {
      printf 'Elapsed (wall clock) time: %02d:%02d:%02d\n' \
        $((ELAPSED_SEC/3600)) $((ELAPSED_SEC%3600/60)) $((ELAPSED_SEC%60))
      echo "Maximum resident set size (kbytes): $PEAK_KB"
    } > ~{out_prefix}.resource_report.txt

    exit "$CB_EXIT"
  >>>

  output {
    File output_h5 = "~{out_prefix}.cellbender.h5"
    File filtered_h5 = "~{out_prefix}.cellbender_filtered.h5"
    File pdf = "~{out_prefix}.cellbender.pdf"
    File cell_barcodes_csv = "~{out_prefix}.cellbender_cell_barcodes.csv"
    File metrics_csv = "~{out_prefix}.cellbender_metrics.csv"
    File posterior_parquet = "~{out_prefix}.cellbender_posterior.parquet"
    File posterior_latents_csv = "~{out_prefix}.cellbender_posterior_latents.csv.gz"
    File posterior_global_latents_json = "~{out_prefix}.cellbender_posterior_global_latents.json"
    File report_html = "~{out_prefix}.cellbender_report.html"
    File log = "~{out_prefix}.log"
    File resource_report = "~{out_prefix}.resource_report.txt"
  }

  runtime {
    docker: docker_image
    cpu: cpu
    memory: "~{memory_gb} GB"
    disks: "local-disk ~{disk_gb} SSD"
    preemptible: 0
  }
}

task OrganizeSample {
  input {
    String sample
    Array[File] h5_files
    Array[File] filtered_h5_files
    Array[File] pdfs
    Array[File] cell_barcodes_csvs
    Array[File] metrics_csvs
    Array[File] posterior_parquets
    Array[File] posterior_latents_csvs
    Array[File] posterior_global_latents_jsons
    Array[File] report_htmls
    Array[File] logs
    Array[File] reports
    String output_gcs_dir
  }

  command <<<
    set -euo pipefail
    gsutil -m cp \
      ~{sep=" " h5_files} ~{sep=" " filtered_h5_files} ~{sep=" " pdfs} \
      ~{sep=" " cell_barcodes_csvs} ~{sep=" " metrics_csvs} ~{sep=" " posterior_parquets} \
      ~{sep=" " posterior_latents_csvs} ~{sep=" " posterior_global_latents_jsons} ~{sep=" " report_htmls} \
      ~{sep=" " logs} ~{sep=" " reports} \
      "~{output_gcs_dir}/~{sample}/"
  >>>

  runtime {
    docker: "google/cloud-sdk:slim"
    cpu: 1
    memory: "2 GB"
    disks: "local-disk 20 HDD"
  }
}
