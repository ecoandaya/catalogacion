#!/usr/bin/env python3
import os
import sys
import csv
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import Iterable, Optional

from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
)

DEFAULT_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".csv", ".json",
    ".docx", ".pptx", ".xlsx", ".rtf"
}


def iter_files(folder: Path, recursive: bool = False, exts: Optional[set[str]] = None) -> Iterable[Path]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    for p in sorted(iterator):
        if p.is_file():
            if exts is None or p.suffix.lower() in exts:
                yield p


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(log_path: Path, record: dict) -> None:
    ensure_parent_dir(log_path)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_csv(csv_path: Path, record: dict) -> None:
    ensure_parent_dir(csv_path)
    file_exists = csv_path.exists()
    fieldnames = [
        "status",
        "timestamp",
        "file_name",
        "file_path",
        "file_size_bytes",
        "remote_file_id",
        "model",
        "response_id",
        "elapsed_seconds",
        "error_type",
        "error",
        "response_text",
    ]
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {k: record.get(k, "") for k in fieldnames}
        writer.writerow(row)


def extract_text_from_response(response) -> str:
    # Ruta preferente del SDK moderno
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()

    # Fallbacks por robustez
    data = None
    try:
        data = response.model_dump()
    except Exception:
        try:
            data = response.to_dict()
        except Exception:
            pass

    if not data:
        return ""

    parts = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                ctype = content.get("type")
                if ctype in ("output_text", "text"):
                    txt = content.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
                    elif isinstance(txt, dict) and "value" in txt:
                        parts.append(str(txt["value"]))
    return "\n".join(p for p in parts if p).strip()


def upload_file(client: OpenAI, filepath: Path):
    with filepath.open("rb") as f:
        return client.files.create(file=f, purpose="user_data")


def delete_remote_file(client: OpenAI, file_id: str) -> None:
    try:
        client.files.delete(file_id)
    except Exception:
        pass


def make_prompt(prompt_template: str, filename: str, filepath: Path) -> str:
    return prompt_template.format(
        filename=filename,
        filepath=str(filepath),
        stem=filepath.stem,
        suffix=filepath.suffix,
    )


def ask_about_file(
    client: OpenAI,
    model: str,
    file_id: str,
    prompt_template: str,
    filename: str,
    filepath: Path,
    extra_instructions: Optional[str] = None,
):
    prompt = make_prompt(prompt_template, filename, filepath)

    kwargs = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_file", "file_id": file_id},
                ],
            }
        ],
    }

    if extra_instructions:
        kwargs["instructions"] = extra_instructions

    return client.responses.create(**kwargs)


def call_with_retries(fn, max_retries: int = 5, base_sleep: float = 2.0):
    """
    Reintenta errores transitorios.
    No reintenta quota insuficiente.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except RateLimitError as e:
            msg = str(e)
            if "insufficient_quota" in msg:
                raise
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = base_sleep * (2 ** (attempt - 1))
            print(f"  Rate limit. Reintento {attempt}/{max_retries} en {sleep_s:.1f}s...", file=sys.stderr)
            time.sleep(sleep_s)

        except (APIConnectionError, APITimeoutError) as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = base_sleep * (2 ** (attempt - 1))
            print(f"  Error de red/timeout. Reintento {attempt}/{max_retries} en {sleep_s:.1f}s...", file=sys.stderr)
            time.sleep(sleep_s)

        except APIStatusError as e:
            # Reintentar 5xx; 4xx no salvo rate limit ya tratado arriba
            status_code = getattr(e, "status_code", None)
            if status_code and 500 <= status_code <= 599:
                attempt += 1
                if attempt > max_retries:
                    raise
                sleep_s = base_sleep * (2 ** (attempt - 1))
                print(f"  Error {status_code}. Reintento {attempt}/{max_retries} en {sleep_s:.1f}s...", file=sys.stderr)
                time.sleep(sleep_s)
                continue
            raise


def parse_extensions(ext_string: Optional[str]) -> Optional[set[str]]:
    if not ext_string:
        return None
    parts = [x.strip().lower() for x in ext_string.split(",") if x.strip()]
    normalized = set()
    for p in parts:
        normalized.add(p if p.startswith(".") else f".{p}")
    return normalized


def print_progress(current: int, total: int, filename: str) -> None:
    width = 28
    ratio = current / total if total else 1
    done = int(width * ratio)
    bar = "#" * done + "-" * (width - done)
    print(f"[{current}/{total}] [{bar}] {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Sube archivos a OpenAI, consulta cada archivo y registra resultados."
    )
    parser.add_argument("folder", help="Carpeta con archivos a procesar")
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="Modelo a usar (por defecto: gpt-4.1)",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Analiza el archivo '{filename}' y responde en español con:\n"
            "1. resumen breve\n"
            "2. 10 etiquetas temáticas\n"
            "3. tipo de documento\n"
            "4. utilidad probable"
        ),
        help="Prompt. Admite {filename}, {filepath}, {stem}, {suffix}",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Instrucciones adicionales para el modelo",
    )
    parser.add_argument(
        "--log",
        default="openai_file_results.jsonl",
        help="Ruta del log JSONL",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Ruta opcional para guardar también CSV",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Procesar subdirectorios recursivamente",
    )
    parser.add_argument(
        "--delete-remote",
        action="store_true",
        help="Eliminar de OpenAI cada archivo tras procesarlo",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Pausa fija entre archivos",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Número máximo de reintentos para errores transitorios",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout del cliente OpenAI en segundos",
    )
    parser.add_argument(
        "--ext",
        default=None,
        help="Filtrar extensiones, separadas por comas. Ej: pdf,txt,docx",
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: define la variable de entorno OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    folder = Path(args.folder).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else None

    if not folder.exists() or not folder.is_dir():
        print(f"Error: la carpeta no existe o no es válida: {folder}", file=sys.stderr)
        sys.exit(1)

    exts = parse_extensions(args.ext)
    if exts is None:
        exts = DEFAULT_EXTENSIONS

    client = OpenAI(api_key=api_key, timeout=args.timeout)

    files = list(iter_files(folder, recursive=args.recursive, exts=exts))
    if not files:
        print("No se encontraron archivos compatibles para procesar.", file=sys.stderr)
        sys.exit(1)

    total = len(files)
    print(f"Procesando {total} archivo(s) en: {folder}")
    print(f"Log JSONL: {log_path}")
    if csv_path:
        print(f"Log CSV:   {csv_path}")

    quota_exhausted = False

    for idx, filepath in enumerate(files, start=1):
        start_ts = time.time()
        remote_file_id = None
        response_id = None

        print_progress(idx, total, filepath.name)

        try:
            uploaded = call_with_retries(
                lambda: upload_file(client, filepath),
                max_retries=args.retries
            )
            remote_file_id = uploaded.id

            response = call_with_retries(
                lambda: ask_about_file(
                    client=client,
                    model=args.model,
                    file_id=remote_file_id,
                    prompt_template=args.prompt,
                    filename=filepath.name,
                    filepath=filepath,
                    extra_instructions=args.instructions,
                ),
                max_retries=args.retries
            )

            response_id = getattr(response, "id", None)
            text = extract_text_from_response(response)

            record = {
                "status": "ok",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_name": filepath.name,
                "file_path": str(filepath),
                "file_size_bytes": filepath.stat().st_size,
                "remote_file_id": remote_file_id,
                "model": args.model,
                "response_id": response_id,
                "elapsed_seconds": round(time.time() - start_ts, 3),
                "error_type": "",
                "error": "",
                "response_text": text,
            }

            append_jsonl(log_path, record)
            if csv_path:
                append_csv(csv_path, record)

            print("  OK")

        except RateLimitError as e:
            msg = str(e)

            if "insufficient_quota" in msg:
                quota_exhausted = True
                human_msg = (
                    "No hay cuota/crédito disponible en la API de OpenAI. "
                    "Revisa facturación y límites del proyecto."
                )
            else:
                human_msg = f"Rate limit excedido: {msg}"

            record = {
                "status": "error",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_name": filepath.name,
                "file_path": str(filepath),
                "file_size_bytes": filepath.stat().st_size if filepath.exists() else "",
                "remote_file_id": remote_file_id,
                "model": args.model,
                "response_id": response_id,
                "elapsed_seconds": round(time.time() - start_ts, 3),
                "error_type": "RateLimitError",
                "error": human_msg,
                "response_text": "",
            }
            append_jsonl(log_path, record)
            if csv_path:
                append_csv(csv_path, record)

            print(f"  ERROR: {human_msg}", file=sys.stderr)

            if quota_exhausted:
                print("Proceso detenido por falta de cuota.", file=sys.stderr)
                break

        except Exception as e:
            record = {
                "status": "error",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_name": filepath.name,
                "file_path": str(filepath),
                "file_size_bytes": filepath.stat().st_size if filepath.exists() else "",
                "remote_file_id": remote_file_id,
                "model": args.model,
                "response_id": response_id,
                "elapsed_seconds": round(time.time() - start_ts, 3),
                "error_type": type(e).__name__,
                "error": str(e),
                "response_text": "",
                "traceback": traceback.format_exc(),
            }
            append_jsonl(log_path, record)
            if csv_path:
                append_csv(csv_path, record)

            print(f"  ERROR: {e}", file=sys.stderr)

        finally:
            if args.delete_remote and remote_file_id:
                delete_remote_file(client, remote_file_id)

        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
