Para rodar este código, você deve ter o Python 3.6+ e o PIP instalados na sua máquina. 

Primeiro, clone o repositório e instale as dependências necessárias rodando:

```bash
git clone https://github.com/thaybaldao/ct213-project.git
cd ct213-project
pip install -r requirements.txt
```

Para salvar e rodar os vídeos do modelo, você deve ter instalado na sua máquina o [FFmpeg](http://ffmpeg.org/). No arquivo `main.py`, há vários parâmetros que podem 
ser customizados para o treinamento, porém os valores padrão já são aqueles que foram usados no projeto. Também é possível escolher o algoritmo (PPO ou A2C) no mesmo arquivo.

O treinamento pode ser feito com:

```bash
python main.py
```

É possível avaliar os modelos com melhores resultados rodando:

```bash
python evaluate.py
```
