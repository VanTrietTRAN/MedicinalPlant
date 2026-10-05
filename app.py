"""Web demo Gradio: nhận diện 206 loài cây dược liệu Việt Nam từ ảnh."""
import html
import json
import logging
import os

import gradio as gr

from medplant import config
from medplant.predictor import Prediction, Predictor, fetch_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

predictor = Predictor()

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
EXAMPLES = sorted(str(p) for p in config.EXAMPLES_DIR.glob("*") if p.suffix.lower() in IMG_EXTS)
VI_NAMES = json.loads(config.VI_NAMES_PATH.read_text(encoding="utf-8"))


def vi_name(name: str) -> str:
    return html.escape(VI_NAMES.get(name, {}).get("vi", ""))


def sci_name(name: str) -> str:
    return f"<i>{html.escape(name.replace('_', ' '))}</i>"


def species(name: str) -> str:
    """Tên Việt (đậm) + tên khoa học (nghiêng, dòng dưới)."""
    vi = vi_name(name)
    return f'<span class="vi">{vi}</span><span class="sci">{sci_name(name)}</span>' if vi else sci_name(name)


def bar_color(p: float) -> str:
    return "#10b981" if p >= 0.85 else "#84cc16" if p >= 0.6 else "#f59e0b"


def topk_rows(topk, highlight_first: bool = True) -> str:
    rows = []
    for rank, (name, p) in enumerate(topk):
        rows.append(f"""
        <div class="row{' first' if highlight_first and rank == 0 else ''}">
            <div class="row-head"><span>{species(name)}</span><b>{p * 100:.1f}%</b></div>
            <div class="bar"><div style="width:{p * 100:.1f}%;background:{bar_color(p)}"></div></div>
        </div>""")
    return "".join(rows)


def final_info(name: str) -> str:
    info = VI_NAMES.get(name, {})
    aka = ", ".join(html.escape(a) for a in info.get("aliases", []))
    out = f'<div class="final-aka">Tên khác: {aka}</div>' if aka else ""
    if info.get("source"):
        out += (f'<a class="final-link" href="{html.escape(info["source"])}" target="_blank" '
                f'rel="noopener">Tra cứu dược liệu ↗</a>')
    return out


def render(pred: Prediction) -> str:
    members = " + ".join(config.MODEL_SPECS[k]["display"] for k in pred.ensemble_members)
    best_name, best_p = pred.ensemble[0]
    cards = "".join(f"""
        <div class="model-card">
            <div class="model-name" style="color:{m.color}">{m.display}
                <span class="acc">test {config.MODEL_SPECS[m.key]['test_top1']:.1f}%</span></div>
            {topk_rows(m.topk)}
        </div>""" for m in pred.models)
    return f"""
    <div class="mp-result">
        <div class="final">
            <div class="final-label">Kết luận (soft-voting: {members})</div>
            <div class="final-pred">{vi_name(best_name) or sci_name(best_name)}</div>
            <div class="final-sci">{sci_name(best_name)}</div>
            {final_info(best_name)}
            <div class="final-conf">Độ tin cậy {best_p * 100:.1f}%</div>
            <div class="final-alt">{topk_rows(pred.ensemble[1:], highlight_first=False)}</div>
        </div>
        <div class="grid">{cards}</div>
        <div class="meta">Thời gian suy luận: {pred.elapsed_ms:.0f} ms (CPU)</div>
    </div>"""


def message(text: str, kind: str = "info") -> str:
    return f'<div class="mp-msg {kind}">{html.escape(text)}</div>'


def run(image, url):
    """Ưu tiên link nếu có; ảnh tải từ link được đưa vào ô ảnh và ô link được xoá."""
    url = (url or "").strip()
    if url:
        try:
            image = fetch_image(url)
        except Exception as e:  # noqa: BLE001 — báo lỗi thân thiện cho người dùng
            log.warning("Không tải được ảnh %s: %s", url, e)
            return gr.skip(), gr.skip(), message(f"Không tải được ảnh từ link: {e}", "error")
    elif image is None:
        return gr.skip(), gr.skip(), message("Hãy tải ảnh lên hoặc dán link ảnh cây thuốc.", "warn")
    return image, "", render(predictor.predict(image))


HERO = f"""
<div class="mp-hero">
    <h1>Nhận Diện Cây Dược Liệu Việt Nam</h1>
    <p>206 loài · {' · '.join(config.MODEL_SPECS[k]['display'] for k in predictor.models)}</p>
</div>"""

DISCLAIMER = """
<div class="mp-footer">Kết quả do mô hình AI dự đoán, chỉ mang tính tham khảo cho mục đích nghiên cứu —
không dùng thay cho chuyên gia khi thu hái hay sử dụng dược liệu.</div>"""

with gr.Blocks(title="Nhận diện cây dược liệu Việt Nam") as demo:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="Ảnh cây thuốc", sources=["upload", "clipboard", "webcam"],
                              height=420)
            url_in = gr.Textbox(label="…hoặc dán link ảnh", placeholder="https://example.com/cay-thuoc.jpg",
                                max_lines=1)
            with gr.Row():
                clear_btn = gr.ClearButton([img_in, url_in], value="Xoá")
                btn = gr.Button("Dự đoán", variant="primary", elem_id="predict-btn", scale=2)
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=img_in, label="Ảnh mẫu")
        with gr.Column(scale=1):
            out_html = gr.HTML(message("Kết quả sẽ hiển thị ở đây."))
    gr.HTML(DISCLAIMER)

    btn.click(run, [img_in, url_in], [img_in, url_in, out_html], concurrency_limit=1)
    url_in.submit(run, [img_in, url_in], [img_in, url_in, out_html], concurrency_limit=1)
    img_in.upload(lambda: "", None, url_in)
    clear_btn.click(lambda: message("Kết quả sẽ hiển thị ở đây."), None, out_html)


if __name__ == "__main__":
    demo.queue(max_size=32).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        theme=gr.themes.Soft(primary_hue="green"),
        css_paths=[config.ROOT / "assets" / "style.css"],
        max_file_size="15mb",
        show_error=True,
    )
