import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_report_pdf(result, filename, upload_time, media_type, out_path):
    """
    Generates a professional forensic PDF report for VeraScope verification results.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
    )
    h2_style = ParagraphStyle(
        "ReportH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    caption_style = ParagraphStyle(
        "ReportCaption",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        alignment=1,
    )

    story = []

    # Header Banner
    story.append(Paragraph("VeraScope &mdash; Forensic Verification Report", title_style))
    story.append(Paragraph("Multi-Modal Deepfake Detection & Explainable AI Evidence Platform", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3b82f6"), spaceAfter=10))

    # Metadata Table
    report_id = os.path.splitext(os.path.basename(out_path))[0]
    faces_summary = f"{result.get('faces_count', 1)} detected ({result.get('flagged_count', 0)} flagged)"

    meta_data = [
        [Paragraph("<b>File Name:</b>", body_style), Paragraph(str(filename), body_style),
         Paragraph("<b>Report ID:</b>", body_style), Paragraph(report_id, body_style)],
        [Paragraph("<b>Media Type:</b>", body_style), Paragraph(media_type.upper(), body_style),
         Paragraph("<b>Analysis Date:</b>", body_style), Paragraph(upload_time, body_style)],
        [Paragraph("<b>Faces Inspected:</b>", body_style), Paragraph(faces_summary, body_style),
         Paragraph("<b>Detector Engine:</b>", body_style), Paragraph("EfficientNet-B4 + Multi-Domain Forensics", body_style)],
    ]
    meta_table = Table(meta_data, colWidths=[95, 165, 95, 165])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Verdict Box
    is_fake = (result.get("prediction") == "Fake")
    verdict_color = colors.HexColor("#dc2626") if is_fake else colors.HexColor("#16a34a")
    verdict_bg = colors.HexColor("#fef2f2") if is_fake else colors.HexColor("#f0fdf4")
    verdict_text = "MANIPULATED / DEEPFAKE DETECTED" if is_fake else "AUTHENTIC / REAL MEDIA"

    verdict_cell = Paragraph(
        f"<font size='13'><b>VERDICT: {verdict_text}</b></font><br/>"
        f"<font size='9.5'>Confidence: <b>{result.get('confidence_pct')}%</b> &nbsp;|&nbsp; "
        f"Authenticity: {result.get('real_pct')}% &nbsp;|&nbsp; Manipulation Risk: {result.get('fake_pct')}%</font>",
        ParagraphStyle("VerdictText", parent=styles["Normal"], textColor=verdict_color, alignment=1, leading=16)
    )
    verdict_table = Table([[verdict_cell]], colWidths=[520])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), verdict_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, verdict_color),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 10))

    # Video details if applicable
    if media_type == "video" and "frames_analyzed" in result:
        story.append(Paragraph("Temporal Video Metrics", h2_style))
        vid_data = [
            [
                Paragraph("<b>Total Sampled Frames:</b>", body_style),
                Paragraph(str(result.get("frames_analyzed", 0)), body_style),
                Paragraph("<b>Frames Flagged Fake:</b>", body_style),
                Paragraph(str(result.get("frames_flagged_fake", 0)), body_style),
            ],
            [
                Paragraph("<b>Key Suspicious Frame:</b>", body_style),
                Paragraph(f"Frame #{result.get('key_frame_index', 'N/A')} ({result.get('key_frame_time', 'N/A')})", body_style),
                Paragraph("<b>Key Frame Fake Score:</b>", body_style),
                Paragraph(f"{result.get('key_frame_fake_pct', 'N/A')}%", body_style),
            ]
        ]
        vid_table = Table(vid_data, colWidths=[130, 130, 130, 130])
        vid_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(vid_table)
        story.append(Spacer(1, 8))

    # Physical & Frequency Forensic Signatures
    metrics = result.get("forensic_metrics")
    if metrics:
        story.append(Paragraph("Multi-Domain Forensic Signal Measurements", h2_style))
        sig_data = [
            [
                Paragraph("<b>2D FFT Radial Spectral Slope:</b>", body_style),
                Paragraph(f"{metrics.get('spectral_slope', 'N/A')} (Baseline: -2.50 to -2.85)", body_style),
                Paragraph("<b>CMOS Sensor Noise Residual:</b>", body_style),
                Paragraph(f"{metrics.get('noise_std', 'N/A')} &sigma; (Baseline: &gt; 2.0 &sigma;)", body_style),
            ],
            [
                Paragraph("<b>Laplacian Texture Variance:</b>", body_style),
                Paragraph(f"{metrics.get('laplacian_variance', 'N/A')} (Baseline: &gt; 50.0)", body_style),
                Paragraph("<b>Error Level Analysis (ELA):</b>", body_style),
                Paragraph(f"{metrics.get('ela_std', 'N/A')} (Baseline: &lt; 1.0)", body_style),
            ]
        ]
        sig_table = Table(sig_data, colWidths=[130, 130, 130, 130])
        sig_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('PADDING', (0, 0), (-1, -1), 3.5),
        ]))
        story.append(sig_table)
        story.append(Spacer(1, 8))

    # Explainability Evidence Visuals
    heatmap_path = result.get("heatmap_path")
    face_path = result.get("face_path")

    story.append(Paragraph("Explainable AI (XAI) Evidence Visuals", h2_style))

    img_elements = []
    if face_path and os.path.exists(face_path):
        img_elements.append(RLImage(face_path, width=155, height=155))
    if heatmap_path and os.path.exists(heatmap_path):
        img_elements.append(RLImage(heatmap_path, width=155, height=155))

    if len(img_elements) == 2:
        img_table = Table([[img_elements[0], img_elements[1]]], colWidths=[260, 260])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 3))
        caption_text = (
            "Figure 1: Extracted Facial Biometric (Left) and Grad-CAM Activation Heatmap Overlay (Right). "
            "Warm color bands (red/orange) highlight highest convolutional salience driving the verdict."
        )
        story.append(Paragraph(caption_text, caption_style))
    elif len(img_elements) == 1:
        story.append(img_elements[0])
        story.append(Spacer(1, 3))
        story.append(Paragraph("Figure 1: Grad-CAM Explainability Heatmap Overlay.", caption_style))
    else:
        story.append(Paragraph("<i>Visual evidence heatmap not available for this run.</i>", body_style))

    story.append(Spacer(1, 8))

    # Detailed Forensic Explanation
    story.append(Paragraph("Forensic Findings & Analysis", h2_style))
    story.append(Paragraph(result.get("explanation", "No detailed explanation provided."), body_style))
    story.append(Spacer(1, 6))

    # Recommendations
    story.append(Paragraph("Operational Recommendation", h2_style))
    story.append(Paragraph(result.get("recommendation", "No specific recommendations provided."), body_style))
    story.append(Spacer(1, 10))

    # Disclaimer Footer
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#94a3b8"), spaceAfter=5))
    disclaimer = (
        "<b>Disclaimer:</b> VeraScope forensic reports are generated using algorithmic deep learning models "
        "(EfficientNet-B4) and multi-domain spectral analysis. This report provides auxiliary evidentiary decision support "
        "and should be considered alongside multi-factor provenance and cryptographic authentication."
    )
    story.append(Paragraph(disclaimer, ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=7, leading=9.5, textColor=colors.HexColor("#94a3b8"))))

    doc.build(story)
    return out_path