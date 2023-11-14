import os

from PyPDF2 import PdfReader, PdfWriter


def extract_pdf_page(input_pdf, page_number, output_pdf):
    pdf_reader = PdfReader(input_pdf)
    pdf_writer = PdfWriter()
    pdf_writer.add_page(pdf_reader.pages[page_number - 1])
    with open(output_pdf, "wb") as f:
        pdf_writer.write(f)


def copy_all_figures():
    os.makedirs("../figures", exist_ok=True)
    for page in [2, 4]:
        extract_pdf_page("archives/perf/plain/1-50/perf-bfgs_cg_newuoa-unconstrained.pdf", page, f"../figures/perf-plain-bfgs_cg_pdfo-50-{page}.pdf")
        extract_pdf_page("archives/perf/nan/1-50/rerun-10_rate-0.01/perf-bfgs_cg_pdfo_pdfo-(no-barrier)-unconstrained.pdf", page, f"../figures/perf-nan-bfgs_cg_pdfo-50-0.01-{page}.pdf")
        extract_pdf_page("archives/perf/nan/1-50/rerun-10_rate-0.05/perf-bfgs_cg_pdfo_pdfo-(no-barrier)-unconstrained.pdf", page, f"../figures/perf-nan-bfgs_cg_pdfo-50-0.05-{page}.pdf")
        extract_pdf_page("archives/perf/noisy/1-50/rerun-10_type-relative_level-1e-08/perf-bfgs_bfgs-adaptive_cg_cg-adaptive_newuoa-unconstrained.pdf", page, f"../figures/perf-noisy-bfgs_cg_pdfo-50-8-{page}.pdf")
        extract_pdf_page("archives/perf/noisy/1-50/rerun-10_type-relative_level-1e-10/perf-bfgs_bfgs-adaptive_cg_cg-adaptive_newuoa-unconstrained.pdf", page, f"../figures/perf-noisy-bfgs_cg_pdfo-50-10-{page}.pdf")


if __name__ == "__main__":
    copy_all_figures()
