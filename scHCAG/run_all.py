
import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch run run2.py on all .h5ad datasets in the data folder."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Folder containing .h5ad datasets. Default: ./data"
    )

    parser.add_argument(
        "--script",
        type=str,
        default="./run2.py",
        help="Path to run2.py. Default: ./run2.py"
    )

    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run run2.py. Default: current Python"
    )

    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=(
            "Optional dataset names to run, without .h5ad suffix. "
            "If omitted, all .h5ad files in data_dir will be used."
        )
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.h5ad",
        help="File pattern used to find datasets. Default: *.h5ad"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./temp",
        help="Output folder used by run2.py. Also used for batch logs. Default: ./temp"
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help=(
            "Skip datasets whose result file already exists in save_dir. "
            "The expected filename is {dataset}_hcag.h5ad."
        )
    )

    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        default=False,
        help="Stop the whole batch if one dataset fails. Default: continue."
    )

    
    args, unknown = parser.parse_known_args()

    
    if unknown and unknown[0] == "--":
        unknown = unknown[1:]

    return args, unknown


def discover_datasets(data_dir, pattern, selected_names=None):
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    if selected_names:
        datasets = []
        for name in selected_names:
            name = Path(name).stem
            file_path = data_dir / f"{name}.h5ad"
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            datasets.append(name)
        return datasets

    files = sorted(data_dir.glob(pattern))
    datasets = [file.stem for file in files if file.is_file()]

    if not datasets:
        raise FileNotFoundError(f"No dataset matched {pattern} in {data_dir}")

    return datasets


def write_summary_header(summary_file):
    summary_file = Path(summary_file)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    if not summary_file.exists():
        with summary_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset",
                "status",
                "return_code",
                "start_time",
                "end_time",
                "elapsed_seconds",
                "command",
                "log_file"
            ])


def append_summary(summary_file, row):
    with Path(summary_file).open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def stream_process(cmd, log_file):
    """
    实时打印 run2.py 输出，并同时写入单数据集日志文件。
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("w", encoding="utf-8", errors="replace") as f:
        f.write("Command:\n")
        f.write(" ".join(cmd) + "\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()

        return process.wait()


def main():
    args, forwarded_args = parse_args()

    data_dir = Path(args.data_dir)
    script = Path(args.script)
    save_dir = Path(args.save_dir)

    if not script.exists():
        raise FileNotFoundError(f"run2.py not found: {script}")

    save_dir.mkdir(parents=True, exist_ok=True)
    batch_log_dir = save_dir / "batch_logs"
    batch_log_dir.mkdir(parents=True, exist_ok=True)

    summary_file = save_dir / "run_all_summary.csv"
    write_summary_header(summary_file)

    datasets = discover_datasets(
        data_dir=data_dir,
        pattern=args.pattern,
        selected_names=args.datasets
    )

    print("=" * 80)
    print(f"Found {len(datasets)} dataset(s) in {data_dir}:")
    for dataset in datasets:
        print(f"  - {dataset}")
    print("=" * 80)

    success_count = 0
    fail_count = 0
    skip_count = 0

    for index, dataset in enumerate(datasets, start=1):
        expected_result = save_dir / f"{dataset}_hcag.h5ad"

        if args.skip_existing and expected_result.exists():
            print(f"\n[{index}/{len(datasets)}] Skip existing dataset: {dataset}")
            skip_count += 1
            append_summary(summary_file, [
                dataset,
                "SKIPPED",
                "",
                "",
                "",
                "",
                "",
                ""
            ])
            continue

        start_dt = datetime.now()
        start_time = time.time()

        timestamp = start_dt.strftime("%Y%m%d_%H%M%S")
        log_file = batch_log_dir / f"{dataset}_{timestamp}.log"

        cmd = [
            args.python,
            str(script),
            "--dataset",
            dataset,
        ] + forwarded_args

        print("\n" + "=" * 80)
        print(f"[{index}/{len(datasets)}] Running dataset: {dataset}")
        print("Command:")
        print(" ".join(cmd))
        print("=" * 80)

        try:
            return_code = stream_process(cmd, log_file)
        except Exception as e:
            return_code = -1
            with log_file.open("a", encoding="utf-8", errors="replace") as f:
                f.write("\n[run_all.py ERROR]\n")
                f.write(str(e) + "\n")
            print(f"\n[ERROR] {dataset}: {e}")

        end_dt = datetime.now()
        elapsed = round(time.time() - start_time, 2)

        if return_code == 0:
            status = "SUCCESS"
            success_count += 1
            print(f"\n[SUCCESS] {dataset}, elapsed={elapsed}s")
        else:
            status = "FAILED"
            fail_count += 1
            print(f"\n[FAILED] {dataset}, return_code={return_code}, elapsed={elapsed}s")
            print(f"Log saved to: {log_file}")

        append_summary(summary_file, [
            dataset,
            status,
            return_code,
            start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            elapsed,
            " ".join(cmd),
            str(log_file)
        ])

        if return_code != 0 and args.stop_on_error:
            print("\nstop_on_error=True, stop batch running.")
            break

    print("\n" + "=" * 80)
    print("Batch finished.")
    print(f"SUCCESS: {success_count}")
    print(f"FAILED : {fail_count}")
    print(f"SKIPPED: {skip_count}")
    print(f"Summary file: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
