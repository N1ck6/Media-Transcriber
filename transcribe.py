"""
Локальная расшифровка аудио/видео с помощью faster-whisper.
"""

import argparse
import os
import sys
import time
import subprocess
import warnings
import logging
import importlib.util
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Подавление шумных предупреждений HuggingFace
# ---------------------------------------------------------------------------
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*symlinks.*")

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"}
SUPPORTED_EXTS = VIDEO_EXTS | AUDIO_EXTS

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def add_nvidia_dll_paths() -> None:
    if sys.platform != "win32":
        return
    try:
        site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
        for name in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc", "cufft", "curand", "cusolver", "cusparse"):
            dll_dir = site_packages / name / "bin"
            if dll_dir.is_dir():
                os.add_dll_directory(str(dll_dir))
                path_env = os.environ.get("PATH", "")
                if str(dll_dir) not in path_env:
                    os.environ["PATH"] = str(dll_dir) + os.pathsep + path_env
    except Exception:
        pass


def check_nvidia_gpu() -> bool:
    try:
        r = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def is_package_installed(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def install_cuda_packages() -> bool:
    print("Устанавливаю CUDA-библиотеки для ускоренного преобразования через GPU (nvidia-cublas-cu12, nvidia-cudnn-cu12)...")
    print("Размер ~800 МБ. Это нужно сделать один раз. Подождите...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        print("CUDA-библиотеки установлены.\n")
        add_nvidia_dll_paths()
        return True
    except Exception as e:
        print(f"Не удалось установить CUDA-пакеты: {e}")
        return False


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def is_supported_media(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS


def check_file_integrity(path: Path) -> bool:
    """
    Проверка целостности:
    1. ffprobe — есть ли валидная длительность
    2. Есть ли хотя бы один аудиопоток
    3. ffmpeg может прочитать первые 1–2 секунды
    """
    # --- 1. Длительность ---
    try:
        cmd_dur = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        r = subprocess.run(cmd_dur, capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return False
        try:
            dur = float(r.stdout.strip() or 0)
        except ValueError:
            return False
        if dur < 0.15:
            return False
    except Exception:
        return False

    # --- 2. Наличие аудиопотока ---
    try:
        cmd_streams = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(path),
        ]
        r = subprocess.run(cmd_streams, capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return False
        # Должна быть хотя бы одна строка "audio"
        if "audio" not in (r.stdout or "").lower():
            return False
    except Exception:
        return False

    # --- 3. Быстрый тест чтения через ffmpeg (первые 2 секунды) ---
    try:
        if sys.platform == "win32":
            null_target = "NUL"
        else:
            null_target = "/dev/null"

        cmd_test = [
            "ffmpeg", "-v", "error",
            "-i", str(path),
            "-t", "2",
            "-f", "null",
            null_target,
        ]
        r = subprocess.run(cmd_test, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return False
    except Exception:
        return False

    return True


def get_duration(path: Path) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def extract_audio(video_path: Path, temp_audio: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "libmp3lame", "-q:a", "4",
        str(temp_audio),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)


def choose_device() -> Tuple[str, str]:
    if not check_nvidia_gpu():
        print("NVIDIA GPU не обнаружена → CPU (медленнее).")
        return "cpu", "int8"

    add_nvidia_dll_paths()
    has_cublas = is_package_installed("nvidia.cublas")
    has_cudnn = is_package_installed("nvidia.cudnn")

    if not (has_cublas and has_cudnn):
        print("GPU найдена, но CUDA-библиотеки отсутствуют.")
        if not install_cuda_packages():
            print("Работаю на CPU.\n")
            return "cpu", "int8"

    try:
        from faster_whisper import WhisperModel
        test = WhisperModel("tiny", device="cuda", compute_type="float16")
        del test
        print("NVIDIA GPU готова → работаю на GPU (значительно быстрее).\n")
        return "cuda", "float16"
    except Exception as e:
        print(f"GPU есть, но CUDA не заработала ({e}). Переключаюсь на CPU.\n")
        return "cpu", "int8"


def write_output(
    segments,
    output_path: Path,
    fmt: str,
    add_timestamps: bool,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        if fmt == "txt":
            for seg in segments:
                text = (seg.text or "").strip()
                if not text:
                    continue
                if add_timestamps:
                    f.write(f"[{seg.start:.1f} → {seg.end:.1f}] {text}\n")
                else:
                    f.write(f"{text}\n")

        elif fmt == "srt":
            for i, seg in enumerate(segments, 1):
                text = (seg.text or "").strip()
                if not text:
                    continue
                start = _sec_to_timestamp(seg.start, srt=True)
                end = _sec_to_timestamp(seg.end, srt=True)
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

        elif fmt == "vtt":
            f.write("WEBVTT\n\n")
            for seg in segments:
                text = (seg.text or "").strip()
                if not text:
                    continue
                start = _sec_to_timestamp(seg.start, srt=False)
                end = _sec_to_timestamp(seg.end, srt=False)
                f.write(f"{start} --> {end}\n{text}\n\n")


def _sec_to_timestamp(sec: float, srt: bool = True) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    if srt:
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def process_file(
    input_path: Path,
    model,
    add_timestamps: bool,
    language: str,
    fmt: str,
    output_dir: Optional[Path],
) -> float:
    start_time = time.time()

    if output_dir:
        output_path = output_dir / (input_path.stem + f".{fmt}")
    else:
        output_path = input_path.with_suffix(f".{fmt}")

    temp_audio: Optional[Path] = None
    audio_path = input_path

    try:
        if is_video(input_path):
            temp_audio = input_path.with_name(input_path.stem + ".temp_audio.mp3")
            if temp_audio.exists() and temp_audio.stat().st_size > 1024:
                print(f"  Найден временный аудиофайл → {temp_audio.name}")
                audio_path = temp_audio
            else:
                print("  Извлекаю аудио...")
                extract_audio(input_path, temp_audio)
                audio_path = temp_audio
                # print(f"  Создан: {temp_audio.name}")

        print("  Расшифровка...")
        segments_gen, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=3,
            vad_filter=True,
        )

        segments = []
        total_duration = getattr(info, "duration", 0.0) or 0.0
        last_percent = -1
        transcribe_start = time.time()

        for seg in segments_gen:
            segments.append(seg)
            if total_duration > 0:
                percent = min(100, int((seg.end / total_duration) * 100))
                if percent != last_percent:
                    elapsed = time.time() - transcribe_start
                    if percent > 0:
                        eta = elapsed * (100 - percent) / percent
                        eta_str = format_duration(eta)
                    else:
                        eta_str = "--:--"
                    bar_len = 30
                    filled = int(bar_len * percent / 100)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"\r  [{bar}] {percent:3d}%  ETA {eta_str}", end="", flush=True)
                    last_percent = percent

        print()
        write_output(segments, output_path, fmt, add_timestamps)
        print(f"  Сохранено: {output_path}")

        if temp_audio and temp_audio.exists():
            try:
                temp_audio.unlink()
                # print("  Временный файл удалён.")
            except OSError:
                print("  Не удалось удалить временный файл.")

    except KeyboardInterrupt:
        print("\n  Прервано.")
        if temp_audio and temp_audio.exists():
            print(f"  Временный файл сохранён: {temp_audio.name}")
        raise
    except Exception as e:
        print(f"\n  Ошибка: {e}")
        if temp_audio and temp_audio.exists():
            print(f"  Временный файл сохранён: {temp_audio.name}")
        return time.time() - start_time

    return time.time() - start_time


def collect_files_from_dir(dir_path: Path) -> List[Path]:
    files = []
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and is_supported_media(p):
            if check_file_integrity(p):
                files.append(p)
            else:
                print(f"Пропуск (повреждён / нет аудио / не читается): {p.name}")
    return files


def collect_files_interactive() -> List[Path]:
    print("Введите абсолютные пути к файлам (по одному на строку).")
    print("Пустая строка — закончить ввод.\n")
    files: List[Path] = []
    while True:
        try:
            raw = input("Путь: ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            break
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            print(f"  Ошибка: не найден → {path}")
            continue
        if not path.is_file():
            print(f"  Ошибка: это не файл → {path}")
            continue
        if not is_supported_media(path):
            print(f"  Ошибка: неподдерживаемый формат → {path.suffix}")
            continue
        if not check_file_integrity(path):
            print(f"  Ошибка: файл повреждён, нет аудио или не читается → {path.name}")
            continue
        files.append(path)
        print(f"  Добавлен: {path.name}")
    return files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Локальная расшифровка аудио/видео (faster-whisper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="*", help="Файлы или ничего (интерактивный режим)")
    parser.add_argument("-t", "--time", action="store_true", help="Таймкоды (для txt)")
    parser.add_argument("--lang", default="ru", help="Язык (по умолчанию ru)")
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
        help="Модель: tiny, base, small, medium, large-v3, turbo (по умолчанию large-v3)",
    )
    parser.add_argument(
        "--format",
        default="txt",
        choices=["txt", "srt", "vtt"],
        help="Формат вывода: txt, srt, vtt (по умолчанию txt)",
    )
    parser.add_argument(
        "--dir",
        metavar="FOLDER",
        help="Обработать все поддерживаемые файлы в указанной папке",
    )
    parser.add_argument(
        "--output-dir",
        metavar="FOLDER",
        help="Папка для сохранения результатов (по умолчанию — рядом с исходником)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    files: List[Path] = []

    if args.dir:
        dir_path = Path(args.dir).expanduser().resolve()
        if not dir_path.is_dir():
            print(f"Ошибка: папка не найдена → {dir_path}")
            return 1
        files = collect_files_from_dir(dir_path)
        if not files:
            print(f"В папке нет поддерживаемых медиафайлов: {dir_path}")
            return 1
    elif args.inputs:
        for item in args.inputs:
            path = Path(item).expanduser().resolve()
            if not path.exists():
                print(f"Пропуск (не найден): {path}")
                continue
            if not path.is_file():
                print(f"Пропуск (не файл): {path}")
                continue
            if not is_supported_media(path):
                print(f"Пропуск (формат): {path}")
                continue
            if not check_file_integrity(path):
                print(f"Пропуск (повреждён / нет аудио / не читается): {path.name}")
                continue
            files.append(path)
    else:
        files = collect_files_interactive()

    if not files:
        print("Нет файлов для обработки.")
        return 1

    output_dir: Optional[Path] = None
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Не удалось создать папку вывода: {e}")
            return 1

    print("\n=== Файлы к обработке ===")
    total_dur = 0.0
    for i, p in enumerate(files, 1):
        dur = get_duration(p)
        total_dur += dur
        print(f"{i}. {p.name}  →  {format_duration(dur)}")
    print(f"Общая длительность: {format_duration(total_dur)}")
    print(f"Формат вывода: {args.format}")
    if output_dir:
        print(f"Папка результатов: {output_dir}")
    print("=" * 40 + "\n")

    device, compute_type = choose_device()

    print(f"Загружаю модель «{args.model}»...")
    print("При первом запуске модель скачается (несколько ГБ). Подождите...\n")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"Не удалось загрузить модель: {e}")
        return 1
    print("Модель готова.\n")

    try:
        for i, path in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {path.name}")
            elapsed = process_file(
                path, model, args.time, args.lang, args.format, output_dir
            )
            print(f"  Время: {format_duration(elapsed)}")
            print("-" * 40)
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return 130

    print("\nГотово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
