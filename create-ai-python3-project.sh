#!/bin/bash

mkdir src/
touch src/main.py

echo "print('Hello, world!')" > src/main.py
echo "numpy
transformers
sentence-transformers
seaborn
torch
torchvision
matplotlib
pandas
scikit-learn
nltk
gensim
tensorflow
keras
opencv-python
fastapi
uvicorn
" > requirements.txt 

echo "#!/bin/bash 

/bin/pip install -r requirements.txt" > install-deps.sh 
chmod +x install-deps.sh

touch run.sh
echo "#!/bin/bash

/bin/python src/main.py" > run.sh
chmod +x run.sh

echo "### Result
* TODO

<img src='' />
" > README.md