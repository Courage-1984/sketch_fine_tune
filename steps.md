instal:

C:\Users\Dieter\AppData\Local\Programs\Python\Python39\python.exe -m venv venv

venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

run:

venv\Scripts\activate

python finetune_sketch_classifier_og.py

