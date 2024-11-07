import PyPDF2
from colorama import Fore, Style

def read_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            total_pages = len(pdf_reader.pages)
            print(f"{Fore.CYAN}Processing {total_pages} pages...{Style.RESET_ALL}")
            
            for i, page in enumerate(pdf_reader.pages):
                if (i + 1) % 50 == 0:
                    print(f"{Fore.CYAN}Processed {i + 1}/{total_pages} pages...{Style.RESET_ALL}")
                text += page.extract_text() + " "
        
        return text.strip()
    except Exception as e:
        print(f"{Fore.RED}Error reading PDF: {e}{Style.RESET_ALL}")
        return None 