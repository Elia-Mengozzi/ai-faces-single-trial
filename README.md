# Single-trial EEG decoding of AI-generated faces

Can a linear decoder distinguish photographs from StyleGAN2-generated faces using a single EEG trial? This project analyses recordings from 26 participants, holding out both participants and presentation sequences. The primary decoder achieved a mean AUC of 0.565.

[run_study.py](run_study.py) is the entry point. [ai_faces_study/](ai_faces_study/) contains the EEG preprocessing, decoding, crossed validation and inference. [study.json](study.json) records the analysis settings. The code uses Python 3.12, NumPy, SciPy, pandas, scikit-learn, MNE-Python, Curry Reader and Matplotlib.

The EEG recordings are held by the University of Milano-Bicocca. Access requests go through the [dataset record](https://doi.org/10.17632/kjv4dckx8m.1).
