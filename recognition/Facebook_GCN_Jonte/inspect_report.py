"""Inspect exported PDF text/images/bounds; render every page for visual review."""

import argparse
import json
import math
from pathlib import Path
import re

import pymupdf

PROJECT = Path(__file__).resolve().parent


def inspect_pdf(pdf, output):
    pdf, output = Path(pdf), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf)
    pages, overflows, feasibility, consultation, empty = [], [], [], [], []
    contact = pymupdf.open()
    sheet = contact.new_page(width=1000, height=math.ceil(len(document) / 4) * 365)
    images = set()
    full_text = []
    for i, page in enumerate(document):
        text = page.get_text()
        full_text.append(text)
        compact = re.sub(r"\s+", "", text).lower()
        if any(line.strip() == "One-page feasibility review" for line in text.splitlines()[:4]):
            feasibility.append(i + 1)
        if "consultationstatus:pending" in compact:
            consultation.append(i + 1)
        if not text.strip() and not page.get_images():
            empty.append(i + 1)
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        x0, y0, x1, y1 = span["bbox"]
                        if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                            overflows.append({"page": i + 1, "text": span["text"], "bbox": span["bbox"]})
        images.update(image[0] for image in page.get_images())
        page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(output / f"page_{i+1:02d}.png")
        small = page.get_pixmap(matrix=pymupdf.Matrix(.35, .35))
        x, y = (i % 4) * 250, (i // 4) * 365
        sheet.insert_text((x + 10, y + 16), f"Page {i+1}", fontsize=12)
        sheet.insert_image(pymupdf.Rect(x + 10, y + 25, x + 240, y + 350), stream=small.tobytes("png"))
        pages.append({"page": i + 1, "text_characters": len(text),
                      "image_count": len(page.get_images()), "link_count": len(page.get_links()),
                      "first_lines": text.splitlines()[:3]})
    sheet.get_pixmap().save(output / "contact_sheet.png")
    text = "\n\n".join(full_text)
    (output / "pdf_text.txt").write_text(text, encoding="utf-8")
    for expected in ("78.95%", "92.48%", "0.920009", "4.184%", "4.802%",
                     "20083", "9781", "3887", "2460", "22056",
                     "Artificial Intelligence Usage Disclosure"):
        if expected not in text:
            raise ValueError(f"Missing expected report content: {expected}")
    if len(feasibility) != 1 or feasibility != consultation:
        raise ValueError(f"Feasibility must occupy one page: start={feasibility}, end={consultation}")
    if overflows or empty:
        raise ValueError(f"PDF layout problem: overflow={overflows}, empty={empty}")
    review = pymupdf.open()
    review.insert_pdf(document, from_page=feasibility[0] - 1, to_page=feasibility[0] - 1)
    review.save(pdf.parent / "COMP3710_Feasibility_Review.pdf")
    report = {"pages": len(document), "unique_images": len(images),
              "feasibility_page": feasibility[0], "overflow_spans": overflows,
              "blank_pages": empty, "page_details": pages}
    (output / "inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=PROJECT / "outputs/submission/COMP3710_Facebook_GCN_Report.pdf")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/submission/inspection")
    args = parser.parse_args()
    inspect_pdf(args.pdf, args.output)


if __name__ == "__main__":
    main()
