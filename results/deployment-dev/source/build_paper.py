"""Generate the preliminary manuscript from measured results and render its PDF."""
from __future__ import annotations

import argparse
import html
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Optional isolated PDF dependencies; does not modify the training environment.
if (ROOT / ".paper-deps").exists():
    sys.path.insert(0, str(ROOT / ".paper-deps"))

import numpy as np
from .analyze import LABELS, clustered, paired, read_rows


def write_draft(folder):
    rows = read_rows(folder / "raw.csv")
    manifest = json.loads((folder / "manifest.json").read_text())
    integrity = json.loads((folder / "summary.json").read_text())
    models = json.loads((folder / "models.json").read_text())
    matched = [r for r in rows if r["family"] != "duffing_mismatch" and r["budget"] == 128
               and r["gauge"] == "original"]
    def value(method, metric, subset=matched):
        return clustered([r for r in subset if r["method"]==method],metric)["mean"]
    q, symbolic = value("query_residual","id_rmse"), value("query_symbolic","id_rmse")
    weight_methods=["head_ridge","jacobian_ridge","normalized_jacobian","finetune_adam"]
    best=min(weight_methods,key=lambda m:value(m,"id_rmse"))
    w=value(best,"id_rmse")
    result_sentence=(f"At 128 calibration queries in the matched families, mean acceleration RMSE "
                     f"was {q:.4f} for a query residual, {symbolic:.4f} for a structured symbolic "
                     f"replacement, and {w:.4f} for the best mean-performing tested weight editor "
                     f"({LABELS[best]}). This last comparator is identified descriptively after evaluation, "
                     "not selected on an independent development set.")
    table=["| Method | Accel. RMSE | Extrap. RMSE | Edit NRMSE | Rollout RMSE |",
           "|---|---:|---:|---:|---:|"]
    for method in manifest["config"]["methods"]:
        table.append("| "+LABELS[method]+" | "+" | ".join(
            f"{value(method,metric):.4f}" for metric in
            ("id_rmse","ood_rmse","edit_nrmse","rollout_rmse"))+" |")
    ps=[]
    for method in weight_methods:
        e=paired(matched,method,"query_residual","id_rmse")
        ps.append(f"{LABELS[method]} minus query residual: {e['mean']:+.4f} "
                  f"(95% seed-cluster interval [{e['low']:+.4f}, {e['high']:+.4f}])")
    gauges=[r for r in rows if r["family"]!="duffing_mismatch" and r["budget"]==128
            and r["gauge"]=="rescaled"]
    gp=[]
    for method in ["query_residual",*weight_methods]:
        gp.append(f"{LABELS[method]}: {value(method,'gauge_disagreement',gauges):.3g}")
    mismatch=[r for r in rows if r["family"]=="duffing_mismatch" and r["budget"]==128
              and r["gauge"]=="original"]
    mismatchtext=("In the misspecified Duffing condition, acceleration RMSE at 128 queries was "
        + "; ".join(f"{value(m,'id_rmse',mismatch):.4f} for {LABELS[m]}"
                    for m in ["query_residual","query_symbolic",best,"oracle_residual"])
        + ". The oracle residual knows the true coefficient decomposition; the other methods do not. "
          "This comparison separates a physical-identification problem from a failure to fit "
          "the synthetic edit targets. It does not establish that weight access resolves misspecification.")
    trainrmse=np.array([m["base_rmse"] for m in models])
    training=(f"Across the 36 base models, held-out pre-intervention acceleration RMSE ranged from "
              f"{trainrmse.min():.4f} to {trainrmse.max():.4f}, with mean {trainrmse.mean():.4f}.")
    cost=(f"The full run completed in {manifest['elapsed_seconds']/60:.2f} minutes on CPU, "
          f"including training and {len(rows):,} condition-method measurements. Mean measured "
          "edit-construction times at 128 queries in the matched original representations were "
          + "; ".join(f"{value(m,'edit_seconds')*1000:.2f} ms for {LABELS[m]}"
                      for m in ["query_residual","head_ridge","jacobian_ridge","normalized_jacobian","finetune_adam"])
          + ". These timings exclude shared coefficient identification and evaluation; they are "
            "single-machine observations, not hardware-independent efficiency estimates. "
          f"There were {int(integrity['nonfinite_rollout_rows'])} nonfinite rollout rows. "
          f"The maximum pre-edit symmetry discrepancy was {integrity['symmetry_max_error']:.3g}.")
    decision=("The current benchmark does not justify a claim that LLM weight-space programming "
              "is needed. Its matched-library setting admits a strong query solution by construction. "
              "The observed representation sensitivity is useful as a control but follows known "
              "optimization geometry. We therefore recommend changing the question before scaling "
              "compute or committing to a paper about programmable world models. This is a no-go "
              "for the original demonstration-style framing, with a conditional go for studying "
              "explicit deployment constraints or information missing from the behavioral interface.")
    tokens=dict(ABSTRACT_RESULTS=result_sentence, MAIN_RESULTS=result_sentence,
                MAIN_TABLE="\n".join(table), PAIRED_RESULTS="Positive paired differences favor the "
                "query residual. Averaging matched families and interventions within each seed, "
                + "; ".join(ps)+". These are exploratory paired contrasts, not multiplicity-adjusted tests.",
                TRAINING_RESULTS=training, NROWS=f"{len(rows):,}",
                GAUGE_RESULTS="After function-preserving rescaling, mean disagreement between "
                "edited predictions and those obtained by editing the original representation was "
                + "; ".join(gp)+". Permutation and rescaling results for every stratum are retained in summary.json.",
                MISMATCH_RESULTS=mismatchtext, COST_RESULTS=cost, DECISION=decision)
    text=(ROOT/"paper/draft.template.md").read_text(encoding="utf-8")
    for key,value_ in tokens.items():
        text=text.replace("{{"+key+"}}",value_)
    if re.search(r"\{\{\w+\}\}",text):
        raise AssertionError("Unfilled manuscript fields")
    (ROOT/"paper/draft.md").write_text(text,encoding="utf-8")
    (ROOT/"docs/decision.md").write_text("# Research decision\n\n"+decision+"\n\n"+
        result_sentence+"\n\n"+training+"\n\n"+cost+"\n\n"+
        "## Next bounded study\n\nRequire deployment inside the original architecture. Compare "
        "query correction plus distillation against direct weight editing on held-out models, "
        "with matched deployment error, runtime, and storage accounting. Establish headroom "
        "before introducing an LLM. If no constraint produces a useful advantage, pivot away "
        "from this intersection rather than enlarging the same benchmark.\n",encoding="utf-8")
    return ROOT/"paper/draft.md"


def render_pdf(source):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                   Table, TableStyle, KeepTogether)
    import matplotlib.mathtext as mathtext
    from matplotlib.font_manager import FontProperties

    fonts=Path("C:/Windows/Fonts")
    for name,file in [("Research","times.ttf"),("ResearchBold","timesbd.ttf"),
                      ("ResearchItalic","timesi.ttf"),("ResearchBoldItalic","timesbi.ttf")]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily("Research",normal="Research",bold="ResearchBold",
                                  italic="ResearchItalic",boldItalic="ResearchBoldItalic")
    output=ROOT/"output/pdf"
    output.mkdir(parents=True,exist_ok=True)
    target=output/"research-draft.pdf"
    styles=getSampleStyleSheet()
    body=ParagraphStyle("ResearchBody",fontName="Research",fontSize=10.5,leading=14,
                        alignment=TA_JUSTIFY,spaceAfter=7)
    heading=ParagraphStyle("ResearchHeading",parent=body,fontName="ResearchBold",fontSize=13,
                           leading=16,spaceBefore=14,spaceAfter=7,keepWithNext=True)
    subheading=ParagraphStyle("ResearchSubHeading",parent=heading,fontSize=11.5,leading=14)
    title=ParagraphStyle("ResearchTitle",parent=body,fontName="ResearchBold",fontSize=20,
                         leading=24,alignment=TA_CENTER,spaceAfter=16)
    small=ParagraphStyle("ResearchSmall",parent=body,fontSize=9,leading=11.5,alignment=0)
    cell=ParagraphStyle("ResearchCell",parent=small,fontSize=8,leading=10,spaceAfter=0)
    doc=SimpleDocTemplate(str(target),pagesize=(612,792),rightMargin=57,leftMargin=57,
                          topMargin=54,bottomMargin=48,title="Before Programming the Weights",
                          author="Exploratory research draft")
    def inline(text):
        text=html.escape(text)
        text=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",text)
        text=re.sub(r"`([^`]+)`",r'<font name="Courier" size="9">\1</font>',text)
        text=re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",r'<link href="\2" color="#164F75">\1</link>',text)
        # Long URLs can wrap without touching the actual hyperlink target.
        text=re.sub(r"(?<![\"=])(https?://[^\s<]+)",lambda m:m.group(0).replace("/","/<wbr/>"),text)
        return text
    lines=source.read_text(encoding="utf-8").splitlines()
    story=[]
    i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]),title))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]),subheading))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]),heading))
        elif line.startswith("$$"):
            equation=line.strip("$ ")
            buffer=io.BytesIO()
            mathtext.math_to_image("$"+equation+"$",buffer,prop=FontProperties(size=13),dpi=220,format="png")
            buffer.seek(0)
            image=Image(buffer)
            scale=min(1.,480/image.imageWidth)
            # mathtext output pixel dimensions at 220 dpi -> physical point size.
            image.drawWidth=image.imageWidth*72/220
            image.drawHeight=image.imageHeight*72/220
            if image.drawWidth>480:
                factor=480/image.drawWidth
                image.drawWidth*=factor
                image.drawHeight*=factor
            image.hAlign="CENTER"
            story.extend([Spacer(1,5),image,Spacer(1,9)])
        elif line.startswith("!["):
            match=re.match(r"!\[([^\]]*)\]\(([^)]+)\)",line)
            path=(source.parent/match.group(2)).resolve()
            image=Image(str(path))
            image.drawHeight*=498/image.drawWidth
            image.drawWidth=498
            story.append(image)
        elif line.startswith("|"):
            data=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                row=[v.strip() for v in lines[i].strip().strip("|").split("|")]
                if not all(re.match(r"^:?-+:?$",v) for v in row):
                    data.append([Paragraph(inline(v),cell) for v in row])
                i+=1
            widths=[138]+[90]*4 if len(data[0])==5 else None
            table=Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EEF2F5")),
                ("LINEBELOW",(0,0),(-1,0),.6,colors.HexColor("#52616D")),
                ("LINEBELOW",(0,-1),(-1,-1),.5,colors.HexColor("#52616D")),
                ("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),5),
                ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
            story.extend([Spacer(1,6),table,Spacer(1,8)])
            continue
        else:
            para=[line]
            while i+1<len(lines) and lines[i+1].strip() and not lines[i+1].startswith(("#","|","![","$$")):
                i+=1
                para.append(lines[i].strip())
            text=" ".join(para)
            style=small if text.startswith(("Figure ","Table ","[1]","[2]","[3]","[4]","[5]","[6]","[7]","[8]")) else body
            story.append(Paragraph(inline(text),style))
        i+=1
    def footer(canvas,doc):
        canvas.saveState()
        canvas.setFont("Research",8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(57,29,"Exploratory draft | November 10, 2026")
        canvas.drawRightString(555,29,str(doc.page))
        canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return target


def validate_pdf(path):
    import pymupdf
    from PIL import Image, ImageOps, ImageDraw
    document=pymupdf.open(path)
    tmp=ROOT/"tmp/pdfs"
    tmp.mkdir(parents=True,exist_ok=True)
    thumbs=[]
    for i,page in enumerate(document):
        pix=page.get_pixmap(matrix=pymupdf.Matrix(1.25,1.25),alpha=False)
        out=tmp/f"page-{i+1:02d}.png"
        pix.save(out)
        image=Image.open(out).convert("RGB")
        image.thumbnail((306,396))
        tile=Image.new("RGB",(326,426),"#dddddd")
        tile.paste(image,((326-image.width)//2,10))
        ImageDraw.Draw(tile).text((10,410),f"Page {i+1}",fill="black")
        thumbs.append(tile)
    cols=3
    sheet=Image.new("RGB",(326*cols,426*((len(thumbs)+cols-1)//cols)),"#bbbbbb")
    for i,image in enumerate(thumbs):
        sheet.paste(image,((i%cols)*326,(i//cols)*426))
    sheet.save(tmp/"contact-sheet.png")
    text="\n".join(page.get_text() for page in document)
    if "{{" in text or "NONFINITE" in text:
        raise AssertionError("Unresolved manuscript fields or unhandled results")
    expected=["Abstract","Experimental design","Research decision","References"]
    if not all(s in text for s in expected):
        raise AssertionError("Missing manuscript sections")
    audit=dict(pages=len(document),words=len(text.split()),rendered_pages=len(thumbs),
               renderer="PyMuPDF; Poppler unavailable",text_checks="passed")
    (ROOT/"output/pdf/validation.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=ROOT/"results/main")
    args=p.parse_args()
    source=write_draft(args.input)
    pdf=render_pdf(source)
    validate_pdf(pdf)
