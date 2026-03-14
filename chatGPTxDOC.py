#
#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import Iterable, Optional

from openai import OpenAI


def iter_files(folder: Path, recursive: bool = False) -> Iterable[Path]:
    if recursive:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                yield p
    else:
        for p in sorted(folder.iterdir()):
            if p.is_file():
                yield p


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def extract_text_from_response(response) -> str:
    """
    Intenta extraer texto útil de la respuesta con varias estrategias,
    para ser robusto frente a cambios menores del SDK.
    """
    # 1) SDK moderno suele exponer output_text
    text = getattr(response, "output_text", None)
    if text:
        return text

    # 2) Inspección manual del objeto serializado
    try:
        data = response.model_dump()
    except Exception:
        try:
            data = response.to_dict()
        except Exception:
            data = None

    if not data:
        return ""

    parts = []

    for item in data.get("output", []):
        # mensajes del asistente
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in ("output_text", "text"):
                    txt = content.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
                    elif isinstance(txt, dict) and "value" in txt:
                        parts.append(str(txt["value"]))

    return "\n".join(p for p in parts if p).strip()


def upload_file(client: OpenAI, filepath: Path):
    with filepath.open("rb") as f:
        # purpose=user_data está soportado por la Files API
        return client.files.create(file=f, purpose="user_data")


def delete_remote_file(client: OpenAI, file_id: str) -> None:
    try:
        client.files.delete(file_id)
    except Exception:
        pass


def ask_about_file(
    client: OpenAI,
    model: str,
    file_id: str,
    prompt_template: str,
    filename: str,
    extra_instructions: Optional[str] = None,
):
    prompt = prompt_template.format(filename=filename)

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


def append_jsonl(log_path: Path, record: dict) -> None:
    ensure_parent_dir(log_path)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sube archivos de una carpeta a OpenAI, consulta cada archivo y registra respuestas."
    )
    parser.add_argument("folder", help="Carpeta con archivos a procesar")
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="Modelo a usar en Responses API (por defecto: gpt-4.1)",
    )
    parser.add_argument(
        "--prompt",
        default="Analiza el archivo '{filename}' y devuelve un resumen claro en español.",
        help="Prompt a usar. Puedes incluir {filename}",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Instrucciones adicionales opcionales",
    )
    parser.add_argument(
        "--log",
        default="openai_file_results.jsonl",
        help="Ruta del log JSONL de salida",
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
        help="Segundos de espera entre archivos",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: define la variable de entorno OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    folder = Path(args.folder).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()

    if not folder.exists() or not folder.is_dir():
        print(f"Error: la carpeta no existe o no es válida: {folder}", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    files = list(iter_files(folder, recursive=args.recursive))
    if not files:
        print("No se encontraron archivos para procesar.", file=sys.stderr)
        sys.exit(1)

    total = len(files)
    print(f"Procesando {total} archivo(s) en: {folder}")
    print(f"Log: {log_path}")

    for idx, filepath in enumerate(files, start=1):
        start_ts = time.time()
        remote_file_id = None

        print(f"[{idx}/{total}] {filepath.name}")

        try:
            uploaded = upload_file(client, filepath)
            remote_file_id = uploaded.id

            response = ask_about_file(
                client=client,
                model=args.model,
                file_id=remote_file_id,
                prompt_template=args.prompt,
                filename=filepath.name,
                extra_instructions=args.instructions,
            )

            text = extract_text_from_response(response)

            record = {
                "status": "ok",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_name": filepath.name,
                "file_path": str(filepath),
                "file_size_bytes": filepath.stat().st_size,
                "remote_file_id": remote_file_id,
                "model": args.model,
                "prompt": args.prompt.format(filename=filepath.name),
                "instructions": args.instructions,
                "response_id": getattr(response, "id", None),
                "response_text": text,
                "elapsed_seconds": round(time.time() - start_ts, 3),
            }
            append_jsonl(log_path, record)

            print("  OK")

        except Exception as e:
            record = {
                "status": "error",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_name": filepath.name,
                "file_path": str(filepath),
                "remote_file_id": remote_file_id,
                "model": args.model,
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(time.time() - start_ts, 3),
            }
            append_jsonl(log_path, record)
            print(f"  ERROR: {e}", file=sys.stderr)

        finally:
            if args.delete_remote and remote_file_id:
                delete_remote_file(client, remote_file_id)

        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
