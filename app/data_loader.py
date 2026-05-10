"""
data_loader.py
PDF ve metin kaynaklarını okuma, temizleme ve belge nesnelerine dönüştürme.
Sonraki adımda doldurulacak.
"""
from pathlib import Path
from typing import List


def load_documents(raw_dir: Path) -> List[dict]:
    """
    data/raw/ klasöründeki PDF ve TXT dosyalarını yükler.

    Args:
        raw_dir: Ham veri klasörü.

    Returns:
        {'source': str, 'content': str} sözlüklerinin listesi.
    """
    raise NotImplementedError("3. Adımda implemente edilecek.")
