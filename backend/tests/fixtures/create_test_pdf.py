"""Create a minimal test PDF for deterministic E2E testing."""
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path

def create_test_pdf(output_path: str):
    """Create a simple test PDF with academic-style content."""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "Test Research Paper")
    
    # Authors
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 96, "Author A, Author B")
    
    # Abstract
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height - 132, "Abstract")
    
    c.setFont("Helvetica", 10)
    abstract_text = (
        "This paper presents a novel approach to machine learning. "
        "We demonstrate significant improvements over baseline methods. "
        "Our method achieves 95% accuracy on standard benchmarks. "
        "However, our approach has limitations including limited dataset diversity "
        "and computational requirements that may affect scalability."
    )
    
    y_pos = height - 156
    for line in abstract_text.split('. '):
        if line:
            c.drawString(72, y_pos, line + ".")
            y_pos -= 15
    
    # Introduction
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y_pos - 24, "Introduction")
    
    c.setFont("Helvetica", 10)
    intro_text = (
        "Machine learning has seen rapid advances in recent years. "
        "Previous work has shown promising results but faces challenges in generalization. "
        "Our research question addresses these limitations."
    )
    y_pos -= 48
    c.drawString(72, y_pos, intro_text)
    
    # Methodology
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y_pos - 36, "Methodology")
    
    c.setFont("Helvetica", 10)
    method_text = (
        "We propose a transformer-based architecture with attention mechanisms. "
        "Our dataset consists of 10,000 samples from diverse sources. "
        "We evaluate using accuracy, precision, recall, and F1-score metrics."
    )
    y_pos -= 60
    c.drawString(72, y_pos, method_text)
    
    # Results
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y_pos - 36, "Results")
    
    c.setFont("Helvetica", 10)
    results_text = (
        "Our method achieves 95% accuracy, outperforming baselines by 10%. "
        "The results show consistent improvements across all evaluation metrics."
    )
    y_pos -= 48
    c.drawString(72, y_pos, results_text)
    
    # Limitations
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y_pos - 36, "Limitations")
    
    c.setFont("Helvetica", 10)
    lim_text = (
        "Our study has several limitations. First, the dataset size is limited which "
        "may affect generalizability. Second, evaluation was restricted to a single domain. "
        "Third, computational cost may limit practical deployment."
    )
    y_pos -= 60
    c.drawString(72, y_pos, lim_text)
    
    # Future Work
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, y_pos - 48, "Future Work")
    
    c.setFont("Helvetica", 10)
    future_text = (
        "Future work should address these limitations by testing on larger datasets "
        "and exploring more efficient architectures."
    )
    y_pos -= 48
    c.drawString(72, y_pos, future_text)
    
    c.save()
    print(f"Created test PDF at {output_path}")

if __name__ == "__main__":
    output = Path(__file__).parent / "test_paper.pdf"
    create_test_pdf(str(output))
