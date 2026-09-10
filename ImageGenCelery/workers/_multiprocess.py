"""Shared multi-process Celery worker runner.

Lets a worker entrypoint (celery_worker.py, celery_support.py, celery_ksampler.py)
run a main worker in the foreground plus optional subprocess workers (via `celery
worker` subprocesses), same behavior as DiffusionService/workers/*.py but shared
across entrypoints instead of tripled.
"""
import os
import sys
import time
import shutil
import signal
import subprocess
import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional
from datetime import datetime


def read_pid_file(pid_file: Path) -> Optional[int]:
    try:
        if pid_file.exists():
            with open(pid_file, 'r') as f:
                return int(f.read().strip())
    except (ValueError, IOError) as e:
        print(f"Warning: Could not read PID file {pid_file}: {e}")
    return None


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_worker_status(pid_path: Path, worker_names: List[str]) -> Dict[str, Dict]:
    status = {}
    for worker_name in worker_names:
        pid_file = pid_path / f"{worker_name}.pid"
        log_file = pid_path.parent / "celery" / f"{worker_name}.log" if "celery" not in str(pid_path) else pid_path / f"{worker_name}.log"

        worker_info = {
            "pid_file": str(pid_file),
            "log_file": str(log_file),
            "pid": None,
            "running": False,
            "status": "stopped"
        }

        pid = read_pid_file(pid_file)
        if pid:
            worker_info["pid"] = pid
            if is_process_running(pid):
                worker_info["running"] = True
                worker_info["status"] = "running"
            else:
                worker_info["status"] = "dead (stale pid file)"

        if log_file.exists():
            mtime = log_file.stat().st_mtime
            worker_info["log_modified"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

        status[worker_name] = worker_info

    return status


def display_worker_status(pid_path: Path, worker_names: List[str]):
    status = get_worker_status(pid_path, worker_names)

    print("\n" + "=" * 80)
    print("CELERY WORKER STATUS")
    print("=" * 80)

    if not status:
        print("No workers configured.")
        return

    for worker_name, info in status.items():
        print(f"\nWorker: {worker_name}")
        print("-" * 80)
        print(f"  Status:    {info['status'].upper()}")
        print(f"  PID:       {info['pid'] if info['pid'] else 'N/A'}")
        print(f"  PID File:  {info['pid_file']}")
        print(f"  Log File:  {info['log_file']}")
        if "log_modified" in info:
            print(f"  Last Log:  {info['log_modified']}")

    print("\n" + "=" * 80)

    running_count = sum(1 for info in status.values() if info["running"])
    total_count = len(status)
    print(f"Summary: {running_count}/{total_count} workers running")
    print("=" * 80 + "\n")


def save_worker_pids(pid_path: Path, worker_names: List[str], output_file: str = "worker_pids.txt"):
    status = get_worker_status(pid_path, worker_names)
    output_path = pid_path / output_file

    try:
        with open(output_path, 'w') as f:
            f.write(f"# Celery Worker PIDs - Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Worker Name | PID | Status\n")
            f.write("-" * 60 + "\n")

            for worker_name, info in status.items():
                pid = info['pid'] if info['pid'] else 'N/A'
                status_str = info['status']
                f.write(f"{worker_name} | {pid} | {status_str}\n")

        print(f"\nWorker PIDs saved to: {output_path}")

    except IOError as e:
        print(f"Error saving PIDs to file: {e}", file=sys.stderr)


def start_main_worker_with_subprocesses(
    celery_app,
    default_queue: str,
    base_dir: Path,
    main_worker_name: str = "worker",
    subprocess_worker_counts: Optional[List[int]] = None,
    subprocess_worker_prefix: str = "worker_sub",
    concurrency: int = 4,
    log_path: Optional[str] = None,
    pid_path: Optional[str] = None,
    pool: str = "threads",
    loglevel: str = "info",
    subprocess_no_logs: bool = False,
    queue_names: Optional[List[str]] = None,
    concurrencies: Optional[List[int]] = None,
    cleanup_subprocesses_on_exit: bool = True,
):
    """Start a main celery worker (foreground) with optional subprocess workers."""
    if subprocess_worker_counts is None:
        subprocess_worker_counts = []

    total_subprocess_workers = sum(subprocess_worker_counts) if subprocess_worker_counts else 0

    if queue_names is None or len(queue_names) == 0:
        main_queue = default_queue
    else:
        main_queue = queue_names[0]

    if concurrencies is None or len(concurrencies) == 0:
        main_concurrency = concurrency
    else:
        main_concurrency = concurrencies[0]

    main_queue = ",".join([q.strip() for q in main_queue.split(",") if q.strip()])

    subprocess_configs = []
    if total_subprocess_workers > 0:
        for group_idx, worker_count in enumerate(subprocess_worker_counts):
            if worker_count <= 0:
                continue

            if queue_names is None or len(queue_names) <= group_idx:
                raise ValueError(
                    f"Not enough queue_names provided. Need at least {group_idx + 1} elements "
                    f"for subprocess groups, but got {len(queue_names) if queue_names else 0}"
                )
            group_queue = queue_names[group_idx]

            if concurrencies is None or len(concurrencies) <= group_idx:
                raise ValueError(
                    f"Not enough concurrencies provided. Need at least {group_idx + 1} elements "
                    f"for subprocess groups, but got {len(concurrencies) if concurrencies else 0}"
                )
            group_concurrency = concurrencies[group_idx]

            group_queue = ",".join([q.strip() for q in group_queue.split(",") if q.strip()])

            subprocess_configs.append({
                'count': worker_count,
                'queue': group_queue,
                'concurrency': group_concurrency,
                'group_idx': group_idx
            })

    worker_specs = []
    worker_counter = 1
    for config in subprocess_configs:
        for _ in range(config['count']):
            worker_name = f"{subprocess_worker_prefix}{worker_counter}"
            worker_specs.append({
                'name': worker_name,
                'queue': config['queue'],
                'concurrency': config['concurrency'],
                'group_idx': config['group_idx']
            })
            worker_counter += 1

    log_path_obj = Path(log_path) if log_path else base_dir / "logs" / "celery"
    pid_path_obj = Path(pid_path) if pid_path else base_dir / "logs" / "pid"

    if pid_path_obj.is_dir():
        shutil.rmtree(pid_path_obj)

    if log_path_obj.is_dir():
        shutil.rmtree(log_path_obj)

    log_path_obj.mkdir(parents=True, exist_ok=True)
    pid_path_obj.mkdir(parents=True, exist_ok=True)

    if total_subprocess_workers > 0:
        print("=" * 80)
        print(f"Starting {total_subprocess_workers} subprocess worker(s) in {len(subprocess_configs)} group(s)...")
        print("=" * 80)

        celery_bin = sys.executable.replace("python", "celery")
        if not os.path.exists(celery_bin):
            celery_bin = "celery"

        worker_names = [spec['name'] for spec in worker_specs]

        print(f"Subprocess workers: {', '.join(worker_names)}")
        print(f"Subprocess logging: {'Disabled (logs to /dev/null)' if subprocess_no_logs else f'Enabled (logs to {log_path_obj})'}")
        print()

        if len(subprocess_configs) > 1:
            print("Group configurations:")
            for config in subprocess_configs:
                print(f"  Group {config['group_idx']+1}: {config['count']} worker(s), "
                      f"concurrency={config['concurrency']}, "
                      f"queue={config['queue'][:60]}..." if len(config['queue']) > 60 else
                      f"queue={config['queue']}")
            print()

        for worker_spec in worker_specs:
            worker_name = worker_spec['name']
            worker_queue = worker_spec['queue']
            worker_concurrency = worker_spec['concurrency']

            if subprocess_no_logs:
                logfile_path = "/dev/null"
            else:
                logfile_path = str(log_path_obj / f"{worker_name}.log")

            pidfile_path = str(pid_path_obj / f"{worker_name}.pid")

            cmd = [
                celery_bin,
                "-A", "celery_app",
                "worker",
                f"-n", f"{worker_name}@%h",
                f"--pidfile={pidfile_path}",
                f"--logfile={logfile_path}",
                f"--loglevel={loglevel}",
                f"--pool={pool}",
                f"--concurrency={worker_concurrency}",
                "-Q", worker_queue
            ]

            print(f"Starting worker {worker_name}:")
            print(f"  Queue: {worker_queue[:100]}..." if len(worker_queue) > 100 else f"  Queue: {worker_queue}")
            print(f"  Concurrency: {worker_concurrency}")
            print(f"  Command: {' '.join(cmd)}")

            try:
                log_files = []
                if subprocess_no_logs:
                    stdout_dest = subprocess.DEVNULL
                    stderr_dest = subprocess.DEVNULL
                else:
                    log_file = open(logfile_path, 'a')
                    log_files.append(log_file)
                    stdout_dest = log_file
                    stderr_dest = log_file

                try:
                    process = subprocess.Popen(
                        cmd,
                        cwd=base_dir,
                        stdout=stdout_dest,
                        stderr=stderr_dest,
                        start_new_session=True
                    )

                    time.sleep(1)

                    if Path(pidfile_path).exists():
                        print(f"  Worker {worker_name} started successfully (PID file created)")
                    else:
                        if process.poll() is None:
                            print(f"  Worker {worker_name} started but PID file not yet created (process running)")
                        else:
                            print(f"  Worker {worker_name} failed to start (exit code: {process.returncode})")
                finally:
                    for log_file in log_files:
                        log_file.close()

            except Exception as e:
                print(f"  Error starting worker {worker_name}: {e}", file=sys.stderr)

            print()

    def cleanup_subprocesses(signum, frame):
        print("\n\n" + "=" * 80)
        print("Shutting down workers...")
        print("=" * 80)

        if total_subprocess_workers > 0 and cleanup_subprocesses_on_exit:
            print(f"\nStopping {total_subprocess_workers} subprocess worker(s)...")

            celery_bin = sys.executable.replace("python", "celery")
            if not os.path.exists(celery_bin):
                celery_bin = "celery"

            worker_names = [spec['name'] for spec in worker_specs]

            for worker_name in worker_names:
                pidfile_path = str(pid_path_obj / f"{worker_name}.pid")

                pid = read_pid_file(Path(pidfile_path))
                if pid and is_process_running(pid):
                    try:
                        os.kill(pid, signal.SIGTERM)
                        print(f"  Sent SIGTERM to worker {worker_name} (PID: {pid})")
                    except Exception as e:
                        print(f"  Error stopping worker {worker_name}: {e}", file=sys.stderr)
                else:
                    cmd = [
                        celery_bin,
                        "-A", "celery_app",
                        "control", "shutdown",
                        f"--destination={worker_name}@%h"
                    ]
                    try:
                        subprocess.run(cmd, cwd=base_dir, capture_output=True, timeout=5)
                    except Exception:
                        pass

            print("Subprocess workers stopped")
        elif total_subprocess_workers > 0:
            print(
                "\nSubprocess cleanup skipped (cleanup_subprocesses_on_exit=False); "
                "subprocess workers keep running."
            )
        else:
            print("\nNo subprocess workers to stop.")

        print("\nShutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_subprocesses)
    signal.signal(signal.SIGTERM, cleanup_subprocesses)

    print("=" * 80)
    print(f"Starting main worker: {main_worker_name} (foreground)")
    print("=" * 80)
    print(f"Pool: {pool}")
    print(f"Concurrency: {main_concurrency}")
    print(f"Log level: {loglevel}")
    print(f"Queues: {main_queue[:100]}..." if len(main_queue) > 100 else f"Queues: {main_queue}")
    if total_subprocess_workers > 0:
        print(f"Subprocess workers: {total_subprocess_workers} in {len(subprocess_configs)} group(s)")
        print(f"Total workers: {total_subprocess_workers + 1} (1 main + {total_subprocess_workers} subprocess)")
        if not cleanup_subprocesses_on_exit:
            print("Subprocess cleanup on SIGINT/SIGTERM: disabled")
    else:
        print(f"Subprocess workers: 0 (main worker only)")
        print(f"Total workers: 1 (main only)")
    print("=" * 80)
    print()

    celery_app.worker_main(
        argv=[
            "worker",
            f"-n", f"{main_worker_name}@%h",
            f"--loglevel={loglevel}",
            f"--pool={pool}",
            f"--concurrency={main_concurrency}",
            "-Q", main_queue
        ]
    )


def build_parser(description: str, default_worker_name: str, default_concurrency: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--action", type=str, default="start", choices=["start", "stop", "status"],
        help="Action: 'start' runs workers (default), 'stop' stops subprocess workers, 'status' shows worker status"
    )
    parser.add_argument(
        "--worker-names", type=str, default=default_worker_name,
        help=f"Main worker name (default: {default_worker_name})"
    )
    parser.add_argument(
        "--num-workers", type=str, default="0",
        help="Subprocess worker counts separated by semicolons (;) for different configuration groups. "
             "Example: '2;3;1' means 2 workers with queue_names[0]/concurrencies[0], "
             "3 with queue_names[1]/concurrencies[1], 1 with queue_names[2]/concurrencies[2]. "
             "Default: '0' (main worker only)."
    )
    parser.add_argument(
        "--concurrency", type=str, default=str(default_concurrency),
        help="Concurrency values separated by semicolons (;) for different workers. "
             "First value is for main worker, rest for subprocesses. "
             f"Default: '{default_concurrency}' (applies to all workers if only one value provided)."
    )
    parser.add_argument("--log-path", type=str, default=None, help="Path to log files directory (default: logs/celery)")
    parser.add_argument("--pid-path", type=str, default=None, help="Path to PID files directory (default: logs/pid)")
    parser.add_argument(
        "--pool", type=str, default="threads", choices=["threads", "prefork", "eventlet", "gevent"],
        help="Worker pool type (default: threads)"
    )
    parser.add_argument(
        "--loglevel", type=str, default="info", choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)"
    )
    parser.add_argument(
        "--subprocess-no-logs", action="store_true", default=False,
        help="Disable logging for subprocess workers (logs discarded to /dev/null). Main worker still logs to stdout."
    )
    parser.add_argument(
        "--no-subprocess-cleanup", action="store_true", default=False,
        help="On SIGINT/SIGTERM, exit the main process without SIGTERM to subprocess workers (they keep running)."
    )
    parser.add_argument(
        "--queue-name", type=str, default=None,
        help="Queue names separated by semicolons (;) for different workers, commas (,) within each worker. "
             "First value is for main worker, rest for subprocesses. "
             "If not provided, uses default worker queues for all workers."
    )
    return parser


def run_worker_cli(
    celery_app,
    get_queue_names: Callable[[], str],
    base_dir: Path,
    default_worker_name: str,
    default_concurrency: int,
):
    """Parse CLI args and dispatch to start/stop/status, same interface across all worker entrypoints."""
    parser = build_parser(
        f"Run Celery workers ({default_worker_name}): main worker (foreground) + optional subprocess workers (background)",
        default_worker_name,
        default_concurrency,
    )
    args = parser.parse_args()

    pid_path_obj = Path(args.pid_path) if args.pid_path else base_dir / "logs" / "pid"
    log_path_obj = Path(args.log_path) if args.log_path else base_dir / "logs" / "celery"

    num_workers_list = None
    if args.num_workers:
        worker_count_strs = [c.strip() for c in args.num_workers.split(";") if c.strip()]
        if worker_count_strs:
            try:
                num_workers_list = [int(c) for c in worker_count_strs]
            except ValueError:
                try:
                    single_count = int(args.num_workers.split(",")[0].strip())
                    num_workers_list = [single_count] if single_count > 0 else []
                    print(f"Warning: Could not parse all num-workers values, using {single_count} for single group")
                except ValueError:
                    print(f"Error: Invalid num-workers format. Using default (no subprocess workers)")
                    num_workers_list = []

    total_subprocesses = sum(num_workers_list) if num_workers_list else 0

    worker_list = []
    if total_subprocesses > 0:
        worker_counter = 1
        if num_workers_list:
            for group_count in num_workers_list:
                for _ in range(group_count):
                    worker_list.append(f"{args.worker_names}_sub{worker_counter}")
                    worker_counter += 1
        else:
            worker_list = [f"{args.worker_names}_sub{i+1}" for i in range(total_subprocesses)]

    if args.action == "status":
        pid_path_obj.mkdir(parents=True, exist_ok=True)
        all_workers = [args.worker_names] + worker_list
        display_worker_status(pid_path_obj, all_workers)
        save_worker_pids(pid_path_obj, all_workers)
        return

    if args.action == "stop":
        if total_subprocesses > 0:
            print("=" * 80)
            print(f"Stopping {total_subprocesses} subprocess worker(s)...")
            print("=" * 80)

            stopped_count = 0
            for worker_name in worker_list:
                pid_file = pid_path_obj / f"{worker_name}.pid"
                pid = read_pid_file(pid_file)

                if pid and is_process_running(pid):
                    try:
                        os.kill(pid, signal.SIGTERM)
                        print(f"  Sent SIGTERM to worker {worker_name} (PID: {pid})")
                        stopped_count += 1
                    except Exception as e:
                        print(f"  Error stopping worker {worker_name}: {e}", file=sys.stderr)
                else:
                    if pid:
                        print(f"  Worker {worker_name} (PID: {pid}) is not running (stale PID file)")
                    else:
                        print(f"  No PID file found for worker {worker_name}")

            if stopped_count > 0:
                print(f"\nStopped {stopped_count}/{total_subprocesses} subprocess worker(s)")
            else:
                print(f"\nNo running subprocess workers found to stop")
        else:
            print("No subprocess workers to stop (only main worker was running).")
        return

    # start mode (default)
    queue_names_list = None
    if args.queue_name:
        queue_configs = [q.strip() for q in args.queue_name.split(";") if q.strip()]
        if queue_configs:
            queue_names_list = queue_configs

    concurrencies_list = None
    if args.concurrency:
        concurrency_strs = [c.strip() for c in args.concurrency.split(";") if c.strip()]
        if concurrency_strs:
            try:
                concurrencies_list = [int(c) for c in concurrency_strs]
            except ValueError:
                try:
                    single_concurrency = int(args.concurrency.split(",")[0].strip())
                    concurrencies_list = [single_concurrency]
                    print(f"Warning: Could not parse all concurrency values, using {single_concurrency} for all workers")
                except ValueError:
                    print(f"Error: Invalid concurrency format. Using default concurrency={default_concurrency}")
                    concurrencies_list = None

    default_concurrency_value = default_concurrency
    if concurrencies_list:
        default_concurrency_value = concurrencies_list[0]
    else:
        try:
            default_concurrency_value = int(args.concurrency.split(",")[0].strip()) if args.concurrency else default_concurrency
        except (ValueError, AttributeError):
            default_concurrency_value = default_concurrency

    if num_workers_list is None:
        try:
            single_count = int(args.num_workers) if args.num_workers else 0
            num_workers_list = [single_count] if single_count > 0 else []
        except (ValueError, AttributeError):
            num_workers_list = []

    start_main_worker_with_subprocesses(
        celery_app=celery_app,
        default_queue=get_queue_names(),
        base_dir=base_dir,
        main_worker_name=args.worker_names,
        subprocess_worker_counts=num_workers_list,
        subprocess_worker_prefix=f"{args.worker_names}_sub",
        concurrency=default_concurrency_value,
        log_path=args.log_path,
        pid_path=args.pid_path,
        pool=args.pool,
        loglevel=args.loglevel,
        subprocess_no_logs=args.subprocess_no_logs,
        queue_names=queue_names_list,
        concurrencies=concurrencies_list,
        cleanup_subprocesses_on_exit=not args.no_subprocess_cleanup,
    )
