import os
import sys
from pathlib import Path
from typing import List

try:
	from pypdf import PdfReader
except Exception as e:
	PdfReader = None

try:
	from openai import OpenAI
except Exception:
	OpenAI = None

ROOT = Path(__file__).resolve().parents[1]
# Default PDF location (will be overridden if auto-discovery finds another one)
PDF_PATH = ROOT / "deep-research" / "anti-patterns-deep-research.pdf"
JAVA_OUT_PATH = ROOT / "deep-research" / "AntiPatterns.java"
RESULT_OUT_PATH = ROOT / "deep-research" / "AntiPatternsRefactored.java"


def read_pdf_all_text(pdf_path: Path) -> str:
	if not pdf_path.exists():
		raise FileNotFoundError(f"PDF not found: {pdf_path}")
	if PdfReader is None:
		raise RuntimeError(
			"pypdf is not installed. Please add 'pypdf' to requirements and reinstall."
		)
	reader = PdfReader(str(pdf_path))
	texts: List[str] = []
	for i, page in enumerate(reader.pages):
		try:
			texts.append(page.extract_text() or "")
		except Exception as e:
			texts.append(f"\n[Page {i+1} extraction error: {e}]\n")
	return "\n\n".join(texts).strip()


def locate_pdf(base: Path) -> Path:
	default = base / "deep-research" / "anti-patterns-deep-research.pdf"
	if default.exists():
		return default
	dr = base / "deep-research"
	if dr.exists():
		for p in dr.rglob("anti-patterns-deep-research.pdf"):
			if p.is_file():
				return p
	return default


def write_java_with_antipatterns(java_path: Path) -> str:
	java_code = """
public class App {
	public static int X = 0;
	public static String data = null;

	public static void main(String[] args){
		App a = new App();
		a.doStuff("hello");
		a.doStuff("world");
		for(int i=0;i<3;i++){
			X = X + i;
		}
		System.out.println("done:" + X);
	}

	public void doStuff(String s){
		if(s!=null && s.length()>0){
			System.out.println("Str:"+s);
			String t = s.trim();
			if(t.equals("hello")){
				System.out.println("Hi!");
			} else if(t.equals("world")){
				System.out.println("Earth!");
			} else {
				System.out.println("???");
			}
		} else {
			System.out.println("bad");
		}

		String u = (s==null?"":s).trim();
		if(u.equals("hello")){
			System.out.println("greetings");
		} else if(u.equals("world")){
			System.out.println("planet");
		} else {
			System.out.println("unknown");
		}

		int r = 42;
		if(r>10){
			for(int j=0;j<5;j++){
				System.out.println("val:"+j);
			}
		}
	}
}
""".strip()
	java_path.parent.mkdir(parents=True, exist_ok=True)
	java_path.write_text(java_code, encoding="utf-8")
	return java_code


def chunk_text(s: str, max_len: int = 8000) -> List[str]:
	if len(s) <= max_len:
		return [s]
	chunks = []
	start = 0
	while start < len(s):
		end = min(start + max_len, len(s))
		chunks.append(s[start:end])
		start = end
	return chunks


def call_openai_refactor(pdf_text: str, java_code: str) -> str:
	if OpenAI is None:
		raise RuntimeError(
			"openai SDK not installed. Please add 'openai' to requirements and reinstall."
		)
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

	client = OpenAI()

	system_msg = (
		"You are a senior Java engineer. Use the provided PDF content on anti-patterns to "
		"identify and fix issues in the Java sample. Prefer clear, idiomatic Java 17+."
	)

	messages: List[dict] = [
		{"role": "system", "content": system_msg},
	]

	pdf_chunks = chunk_text(pdf_text, max_len=8000)
	for i, ch in enumerate(pdf_chunks, start=1):
		messages.append(
			{
				"role": "user",
				"content": f"PDF content part {i}/{len(pdf_chunks)}:\n" + ch,
			}
		)

	messages.append(
		{
			"role": "user",
			"content": (
				"Here is the Java code with anti-patterns. Refactor it to remove the anti-patterns "
				"based on the PDF guidance. Return ONLY a single Java file in a code block, "
				"then a short bullet list of key changes.\n\n" 
				+ "```java\n" + java_code + "\n```"
			),
		}
	)

	resp = client.chat.completions.create(
		model="gpt-5-nano",
		messages=messages,
		temperature=0.2,
	)

	content = resp.choices[0].message.content if resp.choices else ""
	if not content:
		raise RuntimeError("Empty response from model.")
	return content


def extract_java_from_markdown(md: str) -> str:
	fence = "```"
	if fence in md:
		parts = md.split(fence)
		for i in range(len(parts) - 1):
			lang = parts[i + 0].strip().lower()
			body = parts[i + 1]
			if body.lstrip().lower().startswith("java\n"):
				return body.split("\n", 1)[1]
	return md


def main() -> int:
	print("[1/5] Reading PDF...", flush=True)
	pdf_path = locate_pdf(ROOT)
	try:
		pdf_text = read_pdf_all_text(pdf_path)
	except Exception as e:
		print(f"ERROR: {e}")
		print("Hint: Ensure the PDF exists under the 'deep-research/' folder. The app searches recursively for 'anti-patterns-deep-research.pdf'.")
		return 2

	print(f"Read {len(pdf_text):,} characters from PDF.", flush=True)

	print("[2/5] Generating Java anti-pattern sample...", flush=True)
	java_code = write_java_with_antipatterns(JAVA_OUT_PATH)
	print(f"Wrote sample to {JAVA_OUT_PATH} ({len(java_code):,} chars).", flush=True)

	print("[3/5] Calling OpenAI (gpt-5-nano) to refactor...", flush=True)
	try:
		resp_text = call_openai_refactor(pdf_text, java_code)
	except Exception as e:
		print(f"ERROR calling OpenAI: {e}")
		return 3

	print("[4/5] Writing refactoring result...", flush=True)
	RESULT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	RESULT_OUT_PATH.write_text(resp_text, encoding="utf-8")

	try:
		refactored_java = extract_java_from_markdown(resp_text)
		if refactored_java and ("class" in refactored_java):
			clean_path = RESULT_OUT_PATH.with_suffix(".refactored.only.java")
			clean_path.write_text(refactored_java, encoding="utf-8")
			print(f"Saved extracted Java to {clean_path}")
	except Exception:
		pass

	print(f"[5/5] Done. See {RESULT_OUT_PATH}", flush=True)
	return 0

if __name__ == "__main__":
	sys.exit(main())
