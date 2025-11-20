import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader

file_path = "The_Advancement_of_Learning.pdf"

loader = PyMuPDFLoader(file_path, mode="page")
docs = loader.load()

print(f"Loaded {len(docs)} pages")

content = ""
for doc in docs:
	content += doc.page_content + "\n\n"

# First, detect and remove page headers/footers
# These often have patterns like: numbers, all caps, short lines, brackets
def is_page_header_footer(line):
	"""Detect if a line is likely a page header or footer."""
	line = line.strip()
	if not line:
		return False
	
	# Very short lines (1-3 words) that are all caps or mostly numbers
	if len(line) < 30:
		# Check if it's mostly numbers or all caps
		if re.match(r'^[\d\s\[\]]+$', line):  # Just numbers, brackets, spaces
			return True
		if line.isupper() and len(line.split()) <= 5:  # All caps, short
			return True
		if re.match(r'^\d+\s*\[.*\]', line):  # Number followed by brackets
			return True
	
	# Lines that are all caps and contain common header/footer patterns
	if line.isupper():
		header_patterns = [
			r'OF THE ADVANCEMENT OF LEARNING',
			r'ADVANCEMENT OF LEARNING',
			r'^\d+\s*\[.*\]',  # Number with brackets
		]
		for pattern in header_patterns:
			if re.search(pattern, line, re.IGNORECASE):
				return True
	
	return False

# Filter out page headers/footers and normalize the text
lines = content.split('\n')
filtered_lines = []
for line in lines:
	line = line.strip()
	if not line:
		filtered_lines.append("")  # Keep blank lines for paragraph detection
		continue
	
	# Skip page headers/footers
	if is_page_header_footer(line):
		continue
	
	filtered_lines.append(line)

# Now join lines intelligently, allowing content to flow across page boundaries
# Strategy: Join all lines first, then split on actual sentence boundaries
# This allows sentences to span across what were page breaks
normalized_text = ""
for line in filtered_lines:
	if not line:
		# Blank lines become double spaces (paragraph markers)
		normalized_text += "  "
	else:
		# Join lines with a space
		normalized_text += " " + line if normalized_text else line

# Now split into paragraphs (double spaces indicate paragraph breaks)
# But also handle single spaces as potential sentence boundaries
paragraphs = re.split(r'\s{2,}', normalized_text)  # Split on 2+ spaces
normalized_paragraphs = [p.strip() for p in paragraphs if p.strip()]

# Fix hyphenated words that were split across lines
# Pattern: word ending with hyphen, followed by space, followed by word continuation
# Example: "com- memorations" -> "commemorations"
fixed_paragraphs = []
for para in normalized_paragraphs:
	# Replace hyphen-space-word pattern with joined word
	# This matches: word characters, hyphen, space(s), word characters
	# But be careful not to break legitimate hyphenated words at line breaks
	fixed_para = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', para)
	fixed_paragraphs.append(fixed_para)

normalized_paragraphs = fixed_paragraphs

# Now split on sentence-ending periods, but be smart about it
# Don't split on periods in abbreviations, decimals, etc.
sentences = []
for para in normalized_paragraphs:
	para = para.strip()
	if not para:
		continue
	
	# Split on periods that are likely sentence endings
	# Pattern: period followed by space and capital letter, or end of string
	# But avoid: single letters (abbreviations), numbers, etc.
	sentence_parts = re.split(r'(?<=\.)(?=\s+[A-Z]|$)', para)
	
	for part in sentence_parts:
		part = part.strip()
		if not part:
			continue
		
		# Filter out very short fragments that are likely not sentences
		# (unless they end with punctuation)
		if len(part) < 3 and not part.endswith(('.', '!', '?')):
			# Try to merge with previous sentence if it exists
			if sentences:
				sentences[-1] += " " + part
			else:
				sentences.append(part)
		else:
			sentences.append(part)

print(f"Total sentences extracted: {len(sentences)}")


# Generate output filename based on input filename
output_path = Path(file_path).with_suffix('.md')

# Save sentences to markdown file (one sentence per line for easier processing)
with open(output_path, 'w', encoding='utf-8') as f:
	for sentence in sentences:
		f.write(sentence + '\n\n')

print(f"Sentences saved to: {output_path.absolute()}")
print(f"Total sentences: {len(sentences)}")
print(f"Total characters: {sum(len(s) for s in sentences)}")