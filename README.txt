Barcode recovery - Phase 1 (EAN-13)

1) python generate_barcodes.py --count 5000 --out data/clean_all
2) python prepare_dataset.py --src data/clean_all --out data
3) python train.py --data data --epochs 30 --out models/unet.pt
4) python evaluate.py --data data --model models/unet.pt
5) streamlit run app.py        (needs models/unet.pt)

Install: pip install -r requirements.txt
