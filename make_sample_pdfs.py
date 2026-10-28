"""Generate three small PDFs with *fictional* facts, so you can verify the answers
come from the documents and not from the LLM's training data."""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

DOCS = {
    "aurora_coffee_handbook.pdf": [
        ("Aurora Coffee Co. Employee Handbook", [
            "Aurora Coffee Co. was founded in 2014 in the town of Brindlewick by Marta Okonkwo-Hale. "
            "The company operates 23 cafes and one central roastery.",
            "Working hours: Full-time baristas work four 10-hour shifts per week. Shifts start at 6:00 AM "
            "at the earliest. Every employee receives a free 250g bag of beans each month.",
        ]),
        ("Time Off and Benefits", [
            "Paid time off: New employees receive 18 days of paid leave per year, increasing to 25 days "
            "after five years of service. Unused leave can be carried over for a maximum of 6 days.",
            "The company runs a 'Bean Fund' that reimburses up to $400 per year for coffee-related "
            "courses, such as latte art workshops or sensory training.",
        ]),
    ],
    "zephyr9_drone_manual.pdf": [
        ("Zephyr-9 Drone User Manual", [
            "The Zephyr-9 is a foldable camera drone weighing 249 grams. Its battery provides up to "
            "34 minutes of flight time under calm conditions.",
            "Maximum transmission range is 6 km. The built-in camera records 4K video at 60 frames per second.",
        ]),
        ("Safety and Troubleshooting", [
            "If the status light blinks amber three times, the drone has lost GPS lock. Move to an open "
            "area and wait up to 90 seconds before takeoff.",
            "Return-to-Home activates automatically when the battery drops to 15 percent. Do not fly in "
            "winds above 38 km/h. To factory reset, hold the power button for 12 seconds.",
        ]),
    ],
    "lunar_gardening_guide.pdf": [
        ("The Lunar Gardening Guide", [
            "Lunar gardening is the practice of timing planting and harvesting around the phases of the "
            "moon. Supporters of the method claim that moisture in the soil responds to lunar cycles.",
            "Leafy crops such as lettuce and spinach are best sown during the waxing moon, between the "
            "new moon and the full moon.",
        ]),
        ("Root Crops and Pest Control", [
            "Root crops like carrots and beetroot should be planted during the waning moon, when "
            "gardeners believe energy moves downward into the roots.",
            "To control slugs, the guide recommends scattering crushed eggshells and watering in the "
            "early morning. The author, Dr. Ilse Varga, reports 40 percent fewer slugs using this method.",
        ]),
    ],
}


def main():
    out = Path("data")
    out.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    for filename, pages in DOCS.items():
        story = []
        for i, (title, paragraphs) in enumerate(pages):
            story.append(Paragraph(title, styles["Heading1"]))
            story.append(Spacer(1, 12))
            for para in paragraphs:
                story.append(Paragraph(para, styles["BodyText"]))
                story.append(Spacer(1, 10))
            if i < len(pages) - 1:
                story.append(PageBreak())
        SimpleDocTemplate(str(out / filename), pagesize=letter).build(story)
        print(f"Created data/{filename}")


if __name__ == "__main__":
    main()
