"""Minimal local web UI for style_gen."""

from __future__ import annotations

import argparse
import html
import mimetypes
from datetime import datetime
from email.parser import BytesParser
from email.policy import default as default_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from inference import generate_handwriting_page

OUTPUT_DIR = Path("outputs")
UPLOAD_DIR = OUTPUT_DIR / "uploads"


def _default_checkpoint() -> str:
    candidates = [
        Path("training/checkpoints/style_gen_best.pt"),
        Path("training/checkpoints/style_gen_final.pt"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return "training/checkpoints/style_gen_best.pt"


def _guess_content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _page(
    *,
    checkpoint: str = "",
    style_image: str = "",
    text: str = "",
    sampler: str = "ddim",
    steps: str = "50",
    line_spacing: str = "24",
    page_margin: str = "24",
    result_path: str = "",
    error: str = "",
) -> str:
    image_html = ""
    if result_path:
        image_html = (
            f'<h2>Generated</h2>'
            f'<p><a href="/{html.escape(result_path)}" target="_blank">{html.escape(result_path)}</a></p>'
            f'<img src="/{html.escape(result_path)}" alt="Generated handwriting" style="max-width:100%;border:1px solid #d0d0d0;border-radius:10px;">'
        )

    error_html = f'<p style="color:#b00020;">{html.escape(error)}</p>' if error else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>style_gen UI</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --card: #fffaf2;
      --ink: #1e1b18;
      --muted: #6a6259;
      --line: #d8ccb8;
      --accent: #1c6b4a;
      --accent-2: #c86c2e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(200,108,46,0.12), transparent 28%),
        radial-gradient(circle at bottom right, rgba(28,107,74,0.12), transparent 28%),
        var(--bg);
    }}
    main {{
      max-width: 1040px;
      margin: 32px auto;
      padding: 0 16px 40px;
    }}
    .hero {{
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 3vw, 3rem);
      letter-spacing: -0.03em;
    }}
    .help {{
      color: var(--muted);
      max-width: 70ch;
      line-height: 1.5;
    }}
    .shell {{
      margin-top: 10px;
      padding: 12px 14px;
      border-radius: 12px;
      background: #201c17;
      color: #f7f2e9;
      overflow: auto;
      font-family: "Courier New", monospace;
      font-size: 14px;
    }}
    form {{
      display: grid;
      gap: 14px;
      padding: 18px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 14px 40px rgba(30, 27, 24, 0.06);
    }}
    label {{
      display: grid;
      gap: 6px;
      font-weight: 600;
    }}
    input, textarea, select, button {{
      width: 100%;
      font: inherit;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
    }}
    textarea {{
      min-height: 180px;
      resize: vertical;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    .dropzone {{
      border: 2px dashed var(--accent-2);
      background: rgba(200, 108, 46, 0.05);
      padding: 18px;
      border-radius: 14px;
      text-align: center;
      color: var(--muted);
    }}
    .dropzone.drag {{
      background: rgba(28, 107, 74, 0.08);
      border-color: var(--accent);
    }}
    .hint {{
      color: var(--muted);
      font-size: 14px;
      font-weight: 400;
    }}
    button {{
      background: linear-gradient(135deg, var(--accent), #13533a);
      color: white;
      border: 0;
      cursor: pointer;
      font-weight: 700;
    }}
    .error {{
      padding: 12px 14px;
      border-radius: 12px;
      background: rgba(176, 0, 32, 0.08);
      color: #8b0020;
      border: 1px solid rgba(176, 0, 32, 0.18);
    }}
    .result {{
      margin-top: 18px;
      padding: 18px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
    }}
    @media (max-width: 720px) {{
      .row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>style_gen</h1>
      <p class="help">Use a trained checkpoint and either upload a handwriting sample or point to a local image path. The model will generate a new page using that style and the text you type.</p>
      <div class="shell">bash run_ui.sh</div>
    </section>
    {f'<div class="error">{html.escape(error)}</div>' if error else ''}
    <form method="post" action="/" enctype="multipart/form-data">
      <label>Checkpoint path
        <input name="checkpoint" value="{html.escape(checkpoint or _default_checkpoint())}" placeholder="training/checkpoints/style_gen_best.pt" required>
      </label>
      <div class="row">
        <label>Style image path
          <input name="style_image" value="{html.escape(style_image)}" placeholder="/absolute/or/relative/path/to/style_sample.png">
          <span class="hint">Use this if the image already exists on disk.</span>
        </label>
        <label>Or upload a style image
          <div class="dropzone" id="dropzone">
            <div>Drop a handwriting image here or choose a file.</div>
            <input id="style_upload" name="style_upload" type="file" accept="image/*">
          </div>
          <span class="hint">Upload is optional. If both are provided, the uploaded image is used.</span>
        </label>
      </div>
      <label>Text to render
        <textarea name="text" placeholder="Write one or more lines here" required>{html.escape(text)}</textarea>
      </label>
      <div class="row">
        <label>Sampler
          <select name="sampler">
            <option value="ddim" {"selected" if sampler == "ddim" else ""}>ddim</option>
            <option value="ddpm" {"selected" if sampler == "ddpm" else ""}>ddpm</option>
          </select>
        </label>
        <label>Steps
          <input name="steps" value="{html.escape(steps)}" placeholder="50">
        </label>
      </div>
      <div class="row">
        <label>Line spacing
          <input name="line_spacing" value="{html.escape(line_spacing)}" placeholder="24">
        </label>
        <label>Page margin
          <input name="page_margin" value="{html.escape(page_margin)}" placeholder="24">
        </label>
      </div>
      <button type="submit">Generate</button>
    </form>
    {f'<section class="result">{image_html}</section>' if image_html else ''}
  </main>
  <script>
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("style_upload");
    ["dragenter", "dragover"].forEach((eventName) => {{
      dropzone.addEventListener(eventName, (event) => {{
        event.preventDefault();
        dropzone.classList.add("drag");
      }});
    }});
    ["dragleave", "drop"].forEach((eventName) => {{
      dropzone.addEventListener(eventName, (event) => {{
        event.preventDefault();
        dropzone.classList.remove("drag");
      }});
    }});
    dropzone.addEventListener("drop", (event) => {{
      const files = event.dataTransfer.files;
      if (files.length) {{
        fileInput.files = files;
      }}
    }});
  </script>
</body>
</html>"""


def _parse_multipart(handler: BaseHTTPRequestHandler, raw_body: bytes) -> dict[str, object]:
    content_type = handler.headers.get("Content-Type", "")
    message = BytesParser(policy=default_policy).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + raw_body
    )

    parsed: dict[str, object] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            parsed[name] = {"filename": filename, "content": payload}
        else:
            parsed[name] = payload.decode("utf-8", errors="replace")
    return parsed


def _read_form(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw_body = handler.rfile.read(content_length)
    content_type = handler.headers.get("Content-Type", "")

    if content_type.startswith("multipart/form-data"):
        return _parse_multipart(handler, raw_body)

    text_body = raw_body.decode("utf-8")
    return {key: values[0] for key, values in parse_qs(text_body).items()}


def _store_uploaded_style(upload: dict[str, bytes | str]) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(str(upload["filename"])).suffix or ".png"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stored_path = UPLOAD_DIR / f"style_upload_{timestamp}{suffix}"
    stored_path.write_bytes(upload["content"])  # type: ignore[arg-type]
    return str(stored_path)


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/outputs/"):
            file_path = Path(parsed.path.lstrip("/"))
            if file_path.is_file():
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", _guess_content_type(file_path))
                self.end_headers()
                self.wfile.write(file_path.read_bytes())
                return
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        body = _page()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_POST(self) -> None:
        form = _read_form(self)

        checkpoint = str(form.get("checkpoint", "")).strip()
        style_image = str(form.get("style_image", "")).strip()
        text = str(form.get("text", ""))
        sampler = str(form.get("sampler", "ddim"))
        steps = str(form.get("steps", "50")).strip()
        line_spacing = str(form.get("line_spacing", "24")).strip()
        page_margin = str(form.get("page_margin", "24")).strip()

        uploaded_style = form.get("style_upload")
        if isinstance(uploaded_style, dict) and uploaded_style.get("content"):
            style_image = _store_uploaded_style(uploaded_style)

        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"ui_generated_{timestamp}.png"
            generate_handwriting_page(
                checkpoint_path=checkpoint,
                style_image_path=style_image,
                text=text,
                output_path=str(output_path),
                sampler=sampler,
                steps=int(steps) if steps else None,
                line_spacing=int(line_spacing) if line_spacing else 24,
                page_margin=int(page_margin) if page_margin else 24,
            )
            body = _page(
                checkpoint=checkpoint,
                style_image=style_image,
                text=text,
                sampler=sampler,
                steps=steps,
                line_spacing=line_spacing,
                page_margin=page_margin,
                result_path=str(output_path),
            )
        except Exception as exc:
            body = _page(
                checkpoint=checkpoint,
                style_image=style_image,
                text=text,
                sampler=sampler,
                steps=steps,
                line_spacing=line_spacing,
                page_margin=page_margin,
                error=str(exc),
            )

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the style_gen local UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"style_gen UI running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
