"""Automated screenshots + short demo recording for docs/ (dev helper).

Drives the real Streamlit UI with a headless browser (chat textarea path):
  1. Q1: RAG definition -> screenshot with the sources expander open
  2. Q2: off-corpus question -> refusal screenshot
  3. webm recording of the whole session -> docs/demo_recording.webm

The app must be running: streamlit run app.py
"""
import shutil
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"
BASE = "http://localhost:8501"

CHAT_BOX = "textarea[data-testid='stChatInputTextArea']"


def send_question(page, question: str) -> None:
    box = page.wait_for_selector(CHAT_BOX, timeout=30_000)
    box.fill(question)
    box.press("Enter")


def wait_for_answers(page, n: int, timeout_s: int = 360) -> None:
    """Wait until n chat messages exist AND the script is not running.

    NB: the spinner inside st.chat_message also renders an stChatMessage
    element, so message count alone lies. Streamlit shows the stop/status
    widget only while a script run is in progress."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = page.evaluate(
            """() => ({
                msgs: document.querySelectorAll("[data-testid='stChatMessage']").length,
                running: !!document.querySelector("[data-testid='stStatusWidget']"),
            })"""
        )
        if state["msgs"] >= n and not state["running"]:
            return
        time.sleep(2)
    raise TimeoutError(f"answer not settled (msgs={state['msgs']}, running={state['running']}) after {timeout_s}s")


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1180, "height": 1400},
            record_video_dir=str(DOCS / "_video_tmp"),
            record_video_size={"width": 1180, "height": 1400},
        )
        page = context.new_page()
        page.goto(BASE, wait_until="domcontentloaded")

        # Q1 + warm-up: RAG definition (this conversation accumulates all Q/As)
        send_question(page, "What is retrieval-augmented generation and why is it useful?")
        wait_for_answers(page, 2, timeout_s=420)  # first one also loads the LLM
        page.screenshot(path=DOCS / "_tmp_q1.png", full_page=True)

        # open the sources expander of the first answer
        details = page.query_selector_all("details")
        if details:
            details[0].click()
            time.sleep(1.5)
        page.screenshot(path=DOCS / "screenshot_demo.png", full_page=True)
        print("[shot] screenshot_demo.png")

        # Q2: off-corpus -> refusal
        send_question(page, "Who won the 2022 FIFA World Cup?")
        wait_for_answers(page, 4)
        time.sleep(2)
        page.screenshot(path=DOCS / "screenshot_refusal.png", full_page=True)
        print("[shot] screenshot_refusal.png")

        # Q3: second good question -> multi-turn end state
        send_question(page, "What does a perceptron do?")
        wait_for_answers(page, 6)
        time.sleep(1.5)
        page.screenshot(path=DOCS / "screenshot_multiturn.png", full_page=True)
        print("[shot] screenshot_multiturn.png")

        # Q4: switch to the MEDICAL domain and ask a clinical question
        page.get_by_text("Medical literature").click()
        deadline = time.time() + 30
        while time.time() < deadline and "pubmed_" not in page.inner_text("body"):
            time.sleep(1)
        send_question(page, "Do SGLT2 inhibitors help in heart failure with preserved ejection fraction?")
        wait_for_answers(page, 2)
        time.sleep(1.5)
        details = page.query_selector_all("details")
        if details:
            details[0].click()
            time.sleep(1)
        page.screenshot(path=DOCS / "screenshot_medical.png", full_page=True)
        print("[shot] screenshot_medical.png")

        context.close()  # flushes the recorded webm
        browser.close()

    vids = sorted((DOCS / "_video_tmp").glob("*.webm"), key=lambda p: p.stat().st_mtime)
    if vids:
        target = DOCS / "demo_recording.webm"
        target.unlink(missing_ok=True)
        vids[-1].rename(target)  # newest = the finished session
        shutil.rmtree(DOCS / "_video_tmp", ignore_errors=True)
        print("[video] demo_recording.webm")


if __name__ == "__main__":
    main()
