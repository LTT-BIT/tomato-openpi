# ruff: noqa: RUF001
"""Loopback-only training dashboard and detached job supervisor (no extra dependencies)."""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import traceback
from urllib.parse import parse_qs
from urllib.parse import urlsplit
import uuid
from zoneinfo import ZoneInfo

from training_configs import common

TERMINAL = {"completed", "failed", "stopped", "interrupted"}
CHILDREN = []


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def ui_root():
    return common.repository_root() / "logs/training_ui"


def atomic_json(path, value):
    from openpi.shared.local_storage import atomic_write_json

    atomic_write_json(path, value)


@contextmanager
def submit_lock():
    ui_root().mkdir(parents=True, exist_ok=True)
    with (ui_root() / ".submit.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def job_directory(job_id):
    if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        raise ValueError("无效的任务 ID")
    return ui_root() / job_id


def process_identity(pid):
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(") ", 1)[1].split()
        return None if text[0] == "Z" else text[19]
    except (OSError, ValueError, IndexError):
        return None


def read_job(job_id):
    path = job_directory(job_id) / "state.json"
    state = json.loads(path.read_text())
    if (
        state["status"] not in TERMINAL
        and state.get("host") == socket.gethostname()
        and state.get("worker_start")
        and process_identity(state["worker_pid"]) != state["worker_start"]
    ):
        child_alive = state.get("child_start") and process_identity(state.get("child_pid")) == state["child_start"]
        state = {
            **state,
            "status": "orphaned" if child_alive else "interrupted",
            "error": "管理进程已退出，训练子进程仍在运行" if child_alive else "后台进程已退出；查看日志后可续训",
            "updated_at": now(),
        }
    return state


def jobs():
    records = [read_job(path.parent.name) for path in ui_root().glob("*/state.json")]
    return sorted(records, key=lambda row: row["created_at"], reverse=True)


def gpu_inventory():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"gpus": [], "error": f"读取 nvidia-smi 失败: {error}"}
    gpus = []
    for row in csv.reader(result.stdout.splitlines()):
        index, device_uuid, name, total, used, util = (v.strip() for v in row)

        def number(value):
            try:
                return int(value)
            except ValueError:
                return None

        gpus.append(
            {
                "index": index,
                "uuid": device_uuid,
                "name": name,
                "total_mb": number(total),
                "used_mb": number(used),
                "utilization": number(util),
            }
        )
    return {"gpus": gpus, "error": None}


def validate_devices(selected, config, *, allow_busy=False):
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ValueError("请选择至少一张 GPU，不能重复")
    inventory = gpu_inventory()
    available = {gpu["uuid"]: gpu for gpu in inventory["gpus"]}
    if any(device not in available for device in selected):
        raise ValueError(inventory["error"] or "选中的 GPU 已不可用，请刷新 GPU 列表")
    if config.batch_size % len(selected) or len(selected) % config.fsdp_devices:
        raise ValueError("batch_size 必须能被 GPU 数量整除；GPU 数量必须能被 fsdp_devices 整除")
    busy = [
        available[device]["index"]
        for device in selected
        if (available[device]["used_mb"] or 0) > 1024 or (available[device]["utilization"] or 0) > 5
    ]
    if busy and not allow_busy:
        raise ValueError(f"GPU {', '.join(busy)} 正在使用；请换卡，或明确勾选允许使用忙碌 GPU")
    for job in jobs():
        if job["status"] not in TERMINAL and set(selected).intersection(job["gpus"]):
            raise ValueError(f"所选 GPU 已被本仓库 UI 任务 {job['exp_name']} 预留")


def build_request(request):
    from scripts import training_ui_config as config_ui

    module, owner, values = request["baseline"], request["owner"], request["values"]
    config = config_ui.apply_overrides(config_ui.source_config(module), values)
    path = config_ui.source_path(module)
    if request.get("source_sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != request["source_sha256"]:
        raise ValueError("来源配置已修改，请重新选择并核对参数")
    if config_ui.changed_parameters(module, values):
        path = config_ui.personal_path(module, owner, config.exp_name)
    if request.get("wandb_mode") not in ("online", "offline"):
        raise ValueError("请选择 W&B online 或 offline")
    if type(request.get("auto_norm", True)) is not bool or type(request.get("allow_busy", False)) is not bool:
        raise ValueError("auto_norm / allow_busy 必须是布尔值")
    stats_batch = request.get("stats_batch_size", 512)
    if type(stats_batch) is not int or stats_batch <= 0:
        raise ValueError("统计 batch 必须是正整数")
    return config, path


def preview(request, *, check_data=False):
    from openpi.training.reproducibility import serialize_config
    from scripts import training_ui_config as config_ui
    from scripts.run_training import preflight

    config, path = build_request(request)
    result = {
        "config_path": str(path),
        "creates_config": bool(config_ui.changed_parameters(request["baseline"], request["values"])),
        "changed_parameters": config_ui.changed_parameters(request["baseline"], request["values"]),
        "checkpoint_dir": str(config.checkpoint_dir),
        "norm_dir": str(Path(config.data.assets.assets_dir) / config.data.assets.asset_id),
        "config": json.loads(serialize_config(config)),
    }
    if check_data:
        result["data"] = preflight(config, require_norm=False)
        norm = Path(result["norm_dir"])
        if (norm / "norm_stats.json").is_file():
            result["data"] = preflight(config)
            result["statistics"] = "已验证，可复用"
        elif norm.exists():
            raise ValueError(f"统计目录已存在但不完整，请选一个新的 asset_id: {norm}")
        else:
            result["statistics"] = (
                "缺少统计，启动时自动计算" if request.get("auto_norm", True) else "缺少统计，启动会失败"
            )
    return result


def launch_worker(state):
    directory = job_directory(state["id"])
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "state.json", state)
    if "values" in state:
        atomic_json(
            directory / "run.json",
            {
                "module": state["module"],
                "config_sha256": state["config_sha256"],
                "values": state["values"],
            },
        )
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    with (directory / "console.log").open("ab", buffering=0) as stream:
        child = subprocess.Popen(
            [
                "bash",
                str(common.repository_root() / "scripts/python.sh"),
                "-u",
                "-m",
                "scripts.training_ui",
                "--worker",
                state["id"],
            ],
            cwd=common.repository_root(),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state.update(worker_pid=child.pid, worker_start=process_identity(child.pid))
    if state["worker_start"] is None:
        raise OSError("后台进程启动失败，请查看 console.log")
    atomic_json(directory / "state.json", state)
    CHILDREN.append(child)
    # Reap completed children while retaining supervision when the UI itself is closed.
    CHILDREN[:] = [process for process in CHILDREN if process.poll() is None]
    (directory / "ready").touch()
    return state


def start_job(request):
    from openpi.training.reproducibility import serialize_config
    from scripts import training_ui_config as config_ui

    config, path = build_request(request)
    with submit_lock():
        validate_devices(request.get("gpus"), config, allow_busy=request.get("allow_busy", False))
        if any(job["status"] not in TERMINAL and job["checkpoint_dir"] == str(config.checkpoint_dir) for job in jobs()):
            raise ValueError("同一实验已有任务正在运行，请更换实验名称")
        if config.checkpoint_dir.exists():
            raise ValueError("实验 checkpoint 已存在；请换实验名，或从任务记录续训")
        creates_config = bool(config_ui.changed_parameters(request["baseline"], request["values"]))
        if creates_config:
            source = config_ui.personal_source(request["baseline"], request["owner"], request["values"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                stream.write(source)
        else:
            source = path.read_text()
        module = ".".join(path.relative_to(common.repository_root()).with_suffix("").parts)
        state = {
            "id": uuid.uuid4().hex,
            "host": socket.gethostname(),
            "status": "queued",
            "stage": "等待数据校验",
            "created_at": now(),
            "updated_at": now(),
            "baseline": request["baseline"],
            "creates_config": creates_config,
            "values": config_ui.config_values(config),
            "resolved_config": json.loads(serialize_config(config)),
            "owner": request["owner"],
            "exp_name": config.exp_name,
            "module": module,
            "config_path": str(path),
            "config_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "checkpoint_dir": str(config.checkpoint_dir),
            "gpus": request["gpus"],
            "wandb_mode": request["wandb_mode"],
            "auto_norm": request.get("auto_norm", True),
            "stats_batch_size": request.get("stats_batch_size", 512),
            "resume": False,
        }
        try:
            return launch_worker(state)
        except OSError as error:
            state.update(status="failed", error=str(error))
            atomic_json(job_directory(state["id"]) / "state.json", state)
            raise


def resume_job(request):
    with submit_lock():
        original = read_job(request["id"])
        if original.get("host") != socket.gethostname():
            raise ValueError("请在原训练服务器的 UI 中恢复任务")
        if original["status"] not in TERMINAL or original["status"] == "completed":
            raise ValueError("只能重试失败、停止或中断的任务")
        path = Path(original["config_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != original["config_sha256"]:
            raise ValueError("配置文件已修改，不能直接续训；请新建实验")
        config = job_config(original)
        if any(job["status"] not in TERMINAL and job["checkpoint_dir"] == original["checkpoint_dir"] for job in jobs()):
            raise ValueError("同一实验已有任务正在运行")
        selected = request.get("gpus", original["gpus"])
        validate_devices(selected, config, allow_busy=request.get("allow_busy", False))
        # A failed preflight has no checkpoint: retry from scratch. Existing checkpoints use strict resume.
        resume = Path(original["checkpoint_dir"]).exists()
        if resume and not any(
            path.is_dir() and path.name.isdigit() for path in Path(original["checkpoint_dir"]).iterdir()
        ):
            raise ValueError("实验目录已创建但尚无 checkpoint，请使用新的实验名称启动")
        state = {
            **original,
            "id": uuid.uuid4().hex,
            "status": "queued",
            "stage": "准备恢复" if resume else "准备重试",
            "created_at": now(),
            "updated_at": now(),
            "resume": resume,
            "gpus": selected,
            "parent_job": original["id"],
        }
        for key in ("error", "worker_pid", "worker_start", "child_pid", "child_start", "returncode"):
            state.pop(key, None)
        return launch_worker(state)


def stop_job(job_id):
    state = read_job(job_id)
    if state.get("host") != socket.gethostname():
        raise ValueError("请在原训练服务器的 UI 中停止任务")
    if state["status"] in TERMINAL:
        raise ValueError("该任务已结束")
    # Only the recorded, still-identical process group from this job may be signalled.
    if (
        state["status"] == "orphaned"
        and process_identity(state["child_pid"]) == state["child_start"]
        and os.getpgid(state["child_pid"]) == state["child_pid"]
    ):
        os.killpg(state["child_pid"], signal.SIGTERM)
    (job_directory(job_id) / "stop").touch()
    return {"message": "已请求停止；保留已有 checkpoint，不保证保存当前尚未完成的步数"}


class StopRequestedError(Exception):
    pass


def run_stage(directory, state, label, command, env):
    state.update(status="running", stage=label, updated_at=now())
    atomic_json(directory / "state.json", state)
    if (directory / "stop").exists():
        raise StopRequestedError
    print(f"\n[{now()}] {label}", flush=True)
    child = subprocess.Popen(
        command, cwd=common.repository_root(), env=env, stdin=subprocess.DEVNULL, start_new_session=True
    )
    state.update(child_pid=child.pid, child_start=process_identity(child.pid))
    atomic_json(directory / "state.json", state)
    try:
        while child.poll() is None:
            if (directory / "stop").exists():
                raise StopRequestedError
            time.sleep(0.3)
        if child.returncode:
            raise RuntimeError(f"{label}失败，退出码 {child.returncode}；请查看下方日志")
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def job_overrides(state):
    if "values" not in state:  # Older UI records remain usable.
        return []
    return ["--ui-run", str(job_directory(state["id"]) / "run.json")]


def job_config(state):
    from scripts.run_training import resolve_experiment

    return resolve_experiment(state["module"], job_overrides(state))


def worker(job_id):
    directory = job_directory(job_id)
    # The creator publishes the PID before releasing this handshake.
    for _ in range(100):
        if (directory / "ready").exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("UI 未完成任务登记")
    state = read_job(job_id)

    def stop_signal(_signum, _frame):
        (directory / "stop").touch()

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    try:
        path = Path(state["config_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != state["config_sha256"]:
            raise ValueError("保存后配置被修改，请新建实验")
        config = job_config(state)
        overrides = job_overrides(state)
        python = ["bash", str(common.repository_root() / "scripts/python.sh"), "-u", "-m"]
        cpu_env = {**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}
        # Metadata validation is in a separate process so stop remains responsive.
        run_stage(
            directory,
            state,
            "校验数据",
            [*python, "scripts.training_ui", "--validate-module", state["module"], *overrides],
            cpu_env,
        )
        norm = Path(config.data.assets.assets_dir) / config.data.assets.asset_id
        if not (norm / "norm_stats.json").is_file():
            if not state["auto_norm"] or state["resume"]:
                raise ValueError("缺少归一化统计；续训不能重新生成统计")
            run_stage(
                directory,
                state,
                "计算归一化统计",
                [
                    *python,
                    "scripts.compute_experiment_norm",
                    state["module"],
                    *overrides,
                    "--stats-batch-size",
                    str(state["stats_batch_size"]),
                ],
                cpu_env,
            )
        run_stage(
            directory,
            state,
            "核对数据与统计",
            [*python, "scripts.run_training", state["module"], *overrides, "--check"],
            cpu_env,
        )
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(state["gpus"]), "PYTHONUNBUFFERED": "1"}
        env.pop("JAX_PLATFORMS", None)
        env.pop("JAX_PLATFORM_NAME", None)
        # Fail if CUDA cannot initialize rather than accidentally training on CPU.
        env["JAX_PLATFORMS"] = "cuda"
        if hashlib.sha256(path.read_bytes()).hexdigest() != state["config_sha256"]:
            raise ValueError("准备过程中配置被修改，请新建实验")
        command = [*python, "scripts.run_training", state["module"], *overrides, "--wandb-mode", state["wandb_mode"]]
        if state["resume"]:
            command.append("--resume")
        run_stage(directory, state, "训练中", command, env)
        state.update(status="completed", stage="训练完成")
    except StopRequestedError:
        state.update(status="stopped", stage="已停止")
    except Exception as error:
        traceback.print_exc()
        state.update(status="failed", error=str(error))
    finally:
        state["updated_at"] = now()
        atomic_json(directory / "state.json", state)


def browse(path):
    directory = Path(path or common.repository_root()).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("目录不存在或不可访问，请填写训练服务器上的路径")
    entries = [
        {"name": child.name, "path": str(child)}
        for child in sorted(directory.iterdir(), key=lambda item: item.name.lower())
        if not child.name.startswith(".") and child.is_dir() and os.access(child, os.R_OK | os.X_OK)
    ]
    return {
        "path": str(directory),
        "parent": str(directory.parent),
        "entries": entries[:500],
        "truncated": len(entries) > 500,
    }


def detect_sampling_groups(request):
    from scripts import training_ui_config as config_ui
    from scripts.training_ui_groups import detect_groups

    config = config_ui.source_config(request["baseline"])
    if getattr(config.data.base_config, "task_view", None) is not None:
        raise ValueError("当前配置有任务或阶段筛选，请按有效训练 episode 手动填写分组；自动读取目前用于全数据训练")
    return detect_groups(request["dataset_root"])


def handler_class(token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *args):
            pass  # Do not log the session URL or credentials.

        def reply(self, value, status=200, content_type="application/json; charset=utf-8"):
            body = (
                value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def authenticated(self):
            return secrets.compare_digest(self.headers.get("X-Training-Token", ""), token)

        def do_GET(self):  # noqa: N802
            url = urlsplit(self.path)
            static = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8"),
            }
            if url.path in static:
                filename, kind = static[url.path]
                return self.reply(
                    (Path(__file__).parent / "training_ui_web" / filename).read_bytes(), content_type=kind
                )
            if not self.authenticated():
                return self.reply({"error": "请使用启动终端中带 token 的完整地址打开页面"}, 403)
            try:
                from scripts import training_ui_config as config_ui

                query = {key: value[0] for key, value in parse_qs(url.query).items()}
                if url.path == "/api/baselines":
                    value = {
                        "baselines": config_ui.baseline_catalog(),
                        "personal": config_ui.personal_catalog(),
                        "repository": str(common.repository_root()),
                        "python": sys.executable,
                    }
                elif url.path == "/api/baseline":
                    value = config_ui.defaults(query["id"])
                elif url.path == "/api/gpus":
                    value = gpu_inventory()
                elif url.path == "/api/browse":
                    value = browse(query.get("path"))
                elif url.path == "/api/jobs":
                    value = jobs()
                elif url.path == "/api/job":
                    value = read_job(query["id"])
                    path = job_directory(query["id"]) / "console.log"
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - 64 * 1024))
                        value["log"] = stream.read().decode(errors="replace")
                else:
                    return self.reply({"error": "接口不存在"}, 404)
                return self.reply(value)
            except (ValueError, KeyError, TypeError, OSError) as error:
                return self.reply({"error": str(error)}, 400)

        def do_POST(self):  # noqa: N802
            if not self.authenticated() or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.reply({"error": "未授权或请求格式不正确"}, 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("请求大小无效")
                request = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path == "/api/preview":
                    result = preview(request)
                elif path == "/api/sampling-groups":
                    result = detect_sampling_groups(request)
                elif path == "/api/check":
                    result = preview(request, check_data=True)
                elif path == "/api/start":
                    result = start_job(request)
                elif path == "/api/resume":
                    result = resume_job(request)
                elif path == "/api/stop":
                    result = stop_job(request["id"])
                else:
                    return self.reply({"error": "接口不存在"}, 404)
                return self.reply(result)
            except Exception as error:
                return self.reply({"error": str(error)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--validate-module", help=argparse.SUPPRESS)
    parser.add_argument("--ui-run", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    common.configure_environment()
    if args.worker:
        worker(args.worker)
        return
    if args.validate_module:
        from scripts.run_training import preflight
        from scripts.run_training import resolve_experiment

        print(
            json.dumps(
                preflight(
                    resolve_experiment(args.validate_module, ["--ui-run", str(args.ui_run)] if args.ui_run else []),
                    require_norm=False,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_class(token))
    server.daemon_threads = True
    print(f"训练 UI：http://127.0.0.1:{server.server_port}/#token={token}", flush=True)
    print(
        "远程使用时，用 VS Code 转发端口，或在本地运行 ssh -N -L 本地端口:127.0.0.1:服务端口 服务器。打开带 token 的地址。",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
