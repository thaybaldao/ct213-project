# Manual do usuário
Para rodar este código, você deve ter o Python 3.6+ e o PIP instalados na sua máquina.

Então, clone o repositório e instale as dependências necessárias rodando:
```bash
git clone https://github.com/thaybaldao/ct213-project.git
cd ct213-project
pip install -r requirements.txt
```

Para rodar o ambiente Lunar Lander também será preciso instalar o *Box2D*:
```bash
sudo apt install python-box2d
```

Ademais, para salvar e rodar os vídeos do modelo, você deve ter instalado na sua máquina o [FFmpeg](http://ffmpeg.org/):
```bash
sudo apt install ffmpeg
```

Executando o arquivo *grid_search.py*, é possível avaliar modelos que utilizam A2C ou PPO com diferentes conjuntos de hiper-parâmetros.
Todos os parâmetros de treinamento que podem ser customizados estão abaixo do `TODO: choose training and evaluating parameters`.
Além disso, é preciso escolher o algoritmo de treinamento (PPO ou A2C) descomentando a linha que contém `algo = ...` para o algoritmo que deseja treinar.
```bash
python grid_search.py
```

Após o término da execução do *script grid_search.py*, na pasta *logs* você poderá visualizar:
- Os gráficos de treinamento de todos os modelos. 
- Os *.jsons* desccrevendo os hiper-parâmetros utilizados em cada modelo. 
- Um arquivo *.txt* com a avaliação de todos os modelos, no qual os modelos estão ordenados do melhor para o pior modelo, de acordo com a métrica de recompensa média obtida ao longo de `n_eval_episodes` episódios de avaliação.
- Vídeo de simulação do *Lunar Lander* para o melhor modelo.

Agora que você já tem uma noção dos modelos com melhores hiper-parâmetros, você pode realizar um novo treinamento, com mais timesteps e com mais episódios de avaliação, executando o *script evaluate.py*.
Todos os parâmetros de treinamento que podem ser customizados estão abaixo do `TODO: choose training and evaluating parameters`.
Além disso, é preciso escolher o algoritmo de treinamento (PPO ou A2C) descomentando a linha que contém `algo = ...` para o algoritmo que deseja treinar.
Não esqueça de colocar o nome dos algoritmos com melhor performance (obtidos por meio do grid_search) no vetor `best_models_names` para que eles sejam avaliados.*
```bash
python evaluate.py
```

*Neste repositório estão armazenados os modelos 'ppo_9' e 'ppo_89' que podem ser utilizados para rodar o *script evaluate.py*.

Após o término da execução do *script evaluate.py*, na pasta *best_logs* você poderá visualizar:
- Os gráficos de treinamento de todos os modelos. 
- Um arquivo com a avaliação de todos os modelos, no qual os modelos estão ordenados do melhor para o pior modelo, de acordo com a métrica de recompensa média obtida ao longo de `n_eval_episodes` episódios de avaliação.
- Vídeo de simulação do *Lunar Lander* para o melhor modelo.
