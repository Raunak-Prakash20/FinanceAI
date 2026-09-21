import re
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class DocumentChunk:
    chunk_id: str
    ticker: str
    form: str
    section: str  # "ITEM_2_MDA" or "ITEM_1A_RISK"
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    keywords: list[str]
    is_new: bool = False
    is_modified: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class SectionChunker:
    """Chunks SEC filing sections with metadata, overlap, and financial keyword tagging."""

    FINANCIAL_KEYWORDS = [
        "revenue", "operating margin", "gross margin", "net income", "diluted eps",
        "ebitda", "operating cash flow", "free cash flow", "guidance", "backlog",
        "inventory", "working capital", "debt", "credit facility", "covenants",
        "customer concentration", "litigation", "headcount", "capex", "impairment",
        "restructuring", "tariffs", "supply chain", "churn", "pricing power",
    ]

    def __init__(self, target_chunk_size: int = 1800, overlap_chars: int = 250):
        self.target_chunk_size = target_chunk_size
        self.overlap_chars = overlap_chars

    def _tag_keywords(self, text: str) -> list[str]:
        lower = text.lower()
        return [kw for kw in self.FINANCIAL_KEYWORDS if kw in lower]

    def _check_diff_status(
        self, text: str, new_paragraphs: Optional[list[str]], modified_paragraphs: Optional[list[str]]
    ) -> tuple[bool, bool]:
        is_new = False
        is_mod = False
        lower = text.lower()
        if new_paragraphs:
            is_new = any(p[:60].lower() in lower for p in new_paragraphs if len(p) >= 60)
        if modified_paragraphs and not is_new:
            is_mod = any(p[:60].lower() in lower for p in modified_paragraphs if len(p) >= 60)
        return is_new, is_mod

    def chunk_section(
        self,
        text: str,
        ticker: str,
        form: str,
        section: str,
        new_paragraphs: Optional[list[str]] = None,
        modified_paragraphs: Optional[list[str]] = None,
    ) -> list[DocumentChunk]:
        """Split section text into overlapping chunks, breaking cleanly at paragraphs or sentences."""
        if not text.strip():
            return []

        # Split into initial paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[DocumentChunk] = []

        current_text = ""
        current_start = 0
        char_cursor = 0
        chunk_idx = 0

        for para in paragraphs:
            # If paragraph itself exceeds target, split by sentences
            if len(para) > self.target_chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
            else:
                sentences = [para]

            for sent in sentences:
                if len(current_text) + len(sent) + 1 > self.target_chunk_size and current_text:
                    chunk_id = f"{ticker}-{form}-{section}-{chunk_idx:03d}"
                    tagged = self._tag_keywords(current_text)
                    is_n, is_m = self._check_diff_status(current_text, new_paragraphs, modified_paragraphs)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            ticker=ticker,
                            form=form,
                            section=section,
                            text=current_text.strip(),
                            chunk_index=chunk_idx,
                            char_start=current_start,
                            char_end=current_start + len(current_text),
                            keywords=tagged,
                            is_new=is_n,
                            is_modified=is_m,
                        )
                    )
                    chunk_idx += 1

                    # Retain overlap from end of current chunk
                    if self.overlap_chars > 0 and len(current_text) > self.overlap_chars:
                        overlap_text = current_text[-self.overlap_chars:]
                        current_start = char_cursor - len(overlap_text)
                        current_text = overlap_text + "\n" + sent
                    else:
                        current_start = char_cursor
                        current_text = sent
                else:
                    if current_text:
                        current_text += "\n" + sent
                    else:
                        current_start = char_cursor
                        current_text = sent

                char_cursor += len(sent) + 1

        if current_text.strip():
            chunk_id = f"{ticker}-{form}-{section}-{chunk_idx:03d}"
            tagged = self._tag_keywords(current_text)
            is_n, is_m = self._check_diff_status(current_text, new_paragraphs, modified_paragraphs)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    ticker=ticker,
                    form=form,
                    section=section,
                    text=current_text.strip(),
                    chunk_index=chunk_idx,
                    char_start=current_start,
                    char_end=current_start + len(current_text),
                    keywords=tagged,
                    is_new=is_n,
                    is_modified=is_m,
                )
            )

        return chunks
