from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_report_pdf(result, filename, upload_time, media_type, out_path):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    story = []

    story.append(Paragraph("VeraScope - Evidence Verification Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"File analyzed: {filename}", styles["Normal"]))
    story.append(Paragraph(f"Media type: {media_type}", styles["Normal"]))
    story.append(Paragraph(f"Upload time: {upload_time}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Verdict: {result['prediction']}", styles["Heading2"]))
    story.append(Paragraph(f"Confidence: {result['confidence_pct']}%", styles["Normal"]))
    story.append(Paragraph(f"Real: {result['real_pct']}% | Fake: {result['fake_pct']}%", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Explanation", styles["Heading2"]))
    story.append(Paragraph(result["explanation"], styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Recommendation", styles["Heading2"]))
    story.append(Paragraph(result["recommendation"], styles["Normal"]))

    doc.build(story)
    return out_path