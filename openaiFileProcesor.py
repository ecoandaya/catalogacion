#!/usr/bin/env python3
import os
import sys
import csv
import json
import time
import argparse
import traceback
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import fitz  # PyMuPDF
from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
)

# Extensiones con lectura local directa
LOCAL_TEXT_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json"}

# Extensiones admitidas por defecto
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
        "source_mode",
        "extraction_method",
        "pages",
        "extracted_chars",
        "text_truncated",
        "needs_ocr",
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


def make_prompt(prompt_template: str, filename: str, filepath: Path) -> str:
    replacements = {
        "{filename}": filename,
        "{filepath}": str(filepath),
        "{stem}": filepath.stem,
        "{suffix}": filepath.suffix,
    }

    prompt = prompt_template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)

    return prompt


def clean_extracted_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_is_usable(text: str, min_chars: int = 300) -> bool:
    return len(text.strip()) >= min_chars


def truncate_text(text: str, max_chars: int = 50000) -> Tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def extract_text_from_pdf(filepath: Path) -> Tuple[str, int]:
    """
    Extrae texto de un PDF usando PyMuPDF.
    Devuelve (texto, num_paginas).
    """
    doc = fitz.open(filepath)
    parts = []
    try:
        page_count = len(doc)
        for page in doc:
            txt = page.get_text("text")
            if txt:
                parts.append(txt)
    finally:
        doc.close()

    text = "\n".join(parts).strip()
    return text, page_count


def read_text_file(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8", errors="replace")


def save_extracted_text(output_dir: Path, filepath: Path, text: str) -> Path:
    ensure_parent_dir(output_dir / "dummy")
    out_name = filepath.name + ".txt"
    out_path = output_dir / out_name
    with out_path.open("w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def extract_text_locally(
    filepath: Path,
    min_chars: int = 300,
    max_chars: int = 50000,
    save_text_dir: Optional[Path] = None,
) -> dict:
    """
    Intenta extraer texto localmente según el tipo de archivo.

    Devuelve un diccionario con:
    - ok
    - text
    - pages
    - extracted_chars
    - text_truncated
    - needs_ocr
    - source_mode
    - extraction_method
    - saved_text_path
    """
    suffix = filepath.suffix.lower()

    result = {
        "ok": False,
        "text": "",
        "pages": "",
        "extracted_chars": 0,
        "text_truncated": False,
        "needs_ocr": False,
        "source_mode": "",
        "extraction_method": "",
        "saved_text_path": "",
    }

    if suffix == ".pdf":
        raw_text, pages = extract_text_from_pdf(filepath)
        cleaned = clean_extracted_text(raw_text)

        result["pages"] = pages
        result["extracted_chars"] = len(cleaned)
        result["source_mode"] = "pdf_local_text"
        result["extraction_method"] = "pymupdf"

        if text_is_usable(cleaned, min_chars=min_chars):
            final_text, was_truncated = truncate_text(cleaned, max_chars=max_chars)
            result["ok"] = True
            result["text"] = final_text
            result["text_truncated"] = was_truncated
            result["needs_ocr"] = False
        else:
            result["ok"] = False
            result["text"] = cleaned
            result["text_truncated"] = False
            result["needs_ocr"] = True

    elif suffix in {".txt", ".md", ".csv", ".json"}:
        raw_text = read_text_file(filepath)
        cleaned = clean_extracted_text(raw_text)
        final_text, was_truncated = truncate_text(cleaned, max_chars=max_chars)

        result["ok"] = text_is_usable(cleaned, min_chars=min_chars)
        result["text"] = final_text
        result["pages"] = ""
        result["extracted_chars"] = len(cleaned)
        result["text_truncated"] = was_truncated
        result["needs_ocr"] = False
        result["source_mode"] = "local_text"
        result["extraction_method"] = "read_text"

    if save_text_dir and result["text"]:
        out_path = save_extracted_text(save_text_dir, filepath, result["text"])
        result["saved_text_path"] = str(out_path)

    return result


def extract_text_from_response(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()

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


def ask_about_text(
    client: OpenAI,
    model: str,
    text: str,
    prompt_template: str,
    filename: str,
    filepath: Path,
    extra_instructions: Optional[str] = None,
):
    prompt = make_prompt(prompt_template, filename, filepath)

    full_input = (
        f"{prompt}\n\n"
        f"=== NOMBRE DEL ARCHIVO ===\n{filename}\n\n"
        f"=== RUTA DEL ARCHIVO ===\n{filepath}\n\n"
        f"=== CONTENIDO DEL DOCUMENTO ===\n{text}"
    )

    kwargs = {
        "model": model,
        "input": full_input,
    }

    if extra_instructions:
        kwargs["instructions"] = extra_instructions

    return client.responses.create(**kwargs)


def upload_file(client: OpenAI, filepath: Path):
    with filepath.open("rb") as f:
        return client.files.create(file=f, purpose="user_data")


def ask_about_uploaded_file(
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


def delete_remote_file(client: OpenAI, file_id: str) -> None:
    try:
        client.files.delete(file_id)
    except Exception:
        pass


def call_with_retries(fn, max_retries: int = 5, base_sleep: float = 2.0):
    """
    Reintenta errores transitorios.
    No reintenta insufficient_quota.
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


def build_record_base(filepath: Path, model: str) -> dict:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": filepath.name,
        "file_path": str(filepath),
        "file_size_bytes": filepath.stat().st_size if filepath.exists() else "",
        "source_mode": "",
        "extraction_method": "",
        "pages": "",
        "extracted_chars": 0,
        "text_truncated": False,
        "needs_ocr": False,
        "saved_text_path": "",
        "remote_file_id": "",
        "model": model,
        "response_id": "",
        "elapsed_seconds": 0.0,
        "error_type": "",
        "error": "",
        "response_text": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Procesa archivos con OpenAI usando extracción local de texto cuando sea posible."
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
        help="Eliminar de OpenAI cada archivo remoto tras procesarlo",
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
    parser.add_argument(
        "--max-chars",
        type=int,
        default=50000,
        help="Máximo de caracteres a enviar cuando se use extracción local",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=300,
        help="Mínimo de caracteres para considerar útil un texto extraído",
    )
    parser.add_argument(
        "--save-extracted-text",
        default=None,
        help="Directorio opcional donde guardar el texto extraído",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="No subir archivos. Si no se puede extraer texto localmente, registrar error.",
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: define la variable de entorno OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    folder = Path(args.folder).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else None
    save_text_dir = Path(args.save_extracted_text).expanduser().resolve() if args.save_extracted_text else None

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
    if save_text_dir:
        print(f"Texto extraído: {save_text_dir}")

    quota_exhausted = False

    for idx, filepath in enumerate(files, start=1):
        start_ts = time.time()
        remote_file_id = ""
        response_id = ""

        print_progress(idx, total, filepath.name)
        record = build_record_base(filepath, args.model)

        try:
            suffix = filepath.suffix.lower()

            # 1) Intentar extracción local cuando proceda
            used_local_extraction = suffix in LOCAL_TEXT_EXTENSIONS

            if used_local_extraction:
                extraction = extract_text_locally(
                    filepath=filepath,
                    min_chars=args.min_chars,
                    max_chars=args.max_chars,
                    save_text_dir=save_text_dir,
                )

                record["source_mode"] = extraction["source_mode"]
                record["extraction_method"] = extraction["extraction_method"]
                record["pages"] = extraction["pages"]
                record["extracted_chars"] = extraction["extracted_chars"]
                record["text_truncated"] = extraction["text_truncated"]
                record["needs_ocr"] = extraction["needs_ocr"]
                record["saved_text_path"] = extraction["saved_text_path"]

                if extraction["ok"]:
                    response = call_with_retries(
                        lambda: ask_about_text(
                            client=client,
                            model=args.model,
                            text=extraction["text"],
                            prompt_template=args.prompt,
                            filename=filepath.name,
                            filepath=filepath,
                            extra_instructions=args.instructions,
                        ),
                        max_retries=args.retries
                    )

                    response_id = getattr(response, "id", "")
                    record["response_id"] = response_id
                    record["response_text"] = extract_text_from_response(response)
                    record["status"] = "ok"

                else:
                    if suffix == ".pdf" and extraction["needs_ocr"]:
                        raise ValueError("PDF sin texto extraíble suficiente; probablemente necesita OCR")
                    raise ValueError("Texto extraído insuficiente o vacío")

            else:
                # 2) Si no hay extracción local, usar subida de archivo salvo que se fuerce text-only
                if args.text_only:
                    raise ValueError("No se puede extraer texto localmente para este tipo y se ha activado --text-only")

                uploaded = call_with_retries(
                    lambda: upload_file(client, filepath),
                    max_retries=args.retries
                )
                remote_file_id = uploaded.id

                record["source_mode"] = "uploaded_file"
                record["extraction_method"] = "openai_file_upload"
                record["remote_file_id"] = remote_file_id

                response = call_with_retries(
                    lambda: ask_about_uploaded_file(
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

                response_id = getattr(response, "id", "")
                record["response_id"] = response_id
                record["response_text"] = extract_text_from_response(response)
                record["status"] = "ok"

            record["elapsed_seconds"] = round(time.time() - start_ts, 3)

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

            record["status"] = "error"
            record["elapsed_seconds"] = round(time.time() - start_ts, 3)
            record["error_type"] = "RateLimitError"
            record["error"] = human_msg
            record["remote_file_id"] = remote_file_id
            record["response_id"] = response_id

            append_jsonl(log_path, record)
            if csv_path:
                append_csv(csv_path, record)

            print(f"  ERROR: {human_msg}", file=sys.stderr)

            if quota_exhausted:
                print("Proceso detenido por falta de cuota.", file=sys.stderr)
                break

        except Exception as e:
            record["status"] = "error"
            record["elapsed_seconds"] = round(time.time() - start_ts, 3)
            record["error_type"] = type(e).__name__
            record["error"] = str(e)
            record["remote_file_id"] = remote_file_id
            record["response_id"] = response_id
            record["traceback"] = traceback.format_exc()

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
