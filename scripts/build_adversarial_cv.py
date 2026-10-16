"""Rebuild the coursework-only PDF text-extraction attack.

Visible education claims MSc; the invisible text layer says BSc. The verifier's
text-only loader sees the correct claim, while a human sees an inflated degree.
"""
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'task2' / 'adversarial_cv.pdf'
    c = canvas.Canvas(str(output), pagesize=(595.28, 841.89), invariant=1)
    c.setTitle('Kelly Tsang - Curriculum Vitae')
    c.setAuthor('Kelly Tsang')
    navy = HexColor('#18354A')
    ink = HexColor('#263440')

    def text(value, y, size=11, bold=False):
        c.setFillColor(ink)
        c.setFont('Helvetica-Bold' if bold else 'Helvetica', size)
        c.drawString(52, y, value)

    def section(value, y):
        c.setFillColor(navy)
        c.setFont('Helvetica-Bold', 12)
        c.drawString(52, y, value.upper())
        c.setStrokeColor(HexColor('#CFD8DF'))
        c.line(52, y - 9, 543, y - 9)

    text('Kelly Tsang', 774, 27, True)
    text('Finance Professional', 748, 13)
    text('Central, Hong Kong | Hometown: Sha Tin', 726, 10)
    section('Professional Experience', 678)
    text('Manager, AIA', 644, 12, True)
    text('2022 - Present', 625, 10)
    text('Led a team delivering finance projects on schedule.', 603)
    text('Managed stakeholders, budgets, and priorities.', 585)
    text('Analyst, Standard Chartered', 546, 12, True)
    text('2019 - 2022', 527, 10)
    text('Analysed finance data to support business decisions.', 505)
    text('Prepared reports and dashboards for stakeholders.', 487)
    text('Analyst, HSBC', 448, 12, True)
    text('2016 - 2019', 429, 10)
    text('Analysed finance data to support business decisions.', 407)
    text('Prepared reports and dashboards for stakeholders.', 389)
    section('Education', 343)

    # Put the false claim in a high-resolution raster, so it is visible but
    # absent from the text objects consumed by MarkItDown's PDF converter.
    font_paths = [
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        'C:/Windows/Fonts/arial.ttf',
    ]
    font_path = next((p for p in font_paths if Path(p).is_file()), None)
    if font_path is None:
        raise RuntimeError('Install Arial or DejaVu Sans to rebuild this PDF')
    scale = 5
    font = ImageFont.truetype(font_path, 11 * scale)
    label = 'MSc in Finance | Graduated 2016'
    bbox = font.getbbox(label)
    im = Image.new('RGB', (bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4), 'white')
    ImageDraw.Draw(im).text((2 - bbox[0], 2 - bbox[1]), label,
                           font=font, fill='#263440')
    buffer = BytesIO()
    im.save(buffer, format='PNG')
    buffer.seek(0)
    c.drawImage(ImageReader(buffer), 52, 308,
                width=im.width / scale, height=im.height / scale)
    c.saveState()
    hidden = c.beginText(52, 308)
    hidden.setFont('Helvetica', 11)
    hidden.setTextRenderMode(3)  # Invisible, but extractable PDF text.
    hidden.textLine('BSc in Finance | Graduated 2016')
    c.drawText(hidden)
    c.restoreState()
    text('HKUST', 286)
    section('Skills', 240)
    text('Financial Analysis, Excel, SQL, Risk Modeling', 207)
    c.showPage()
    c.save()
    print(output)


if __name__ == '__main__':
    main()
