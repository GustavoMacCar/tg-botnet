# Como executar este projeto:

Faça o download dos repositórios presentes em:

https://github.com/GustavoMacCar/tg-botnet

https://github.com/GustavoMacCar/EvoloPy-FS/tree/main

Substitua na linha 19 do arquivo main.py do projeto EvoloPY-FS o caminho onde foi salvo o projeto tg-botnet.

Substitua nas linhas 5 e 7 do arquivo tg.sh do projeto EvoloPy-FS o caminho onde foi salvo o projeto tg-botnet e o caminho onde foi salvo o projeto EvoloPY-FS, respectivamente.

Num sistema baseado em Unix, execute o comando **chmod u+x tg.sh** no diretorio onde foi salvo o arquivo tg.sh. Para executar o projeto, agora rode o comando **./tg.sh**. Pode ser necessário instalar dependências. As dependências utilizadas foram SciKitLearn, Numpy e Pandas, assim como suas sub dependências.

# Como manusear o programa:

## Para executar o modelo sem nenhuma otimização:

No diretório do projeto tg-botnet, apague todo o conteúdo dos arquivos first_optimizer.csv, second_optimizer.csv, third_optmizer.csv e features.csv.
Na linha 17 do arquivo main.py do projeto tg-botnet, faça a leitura do arquivo csv que deseja usar para alimentar o modelo.
Então execute o seguinte comando **python3 main.py arg ctu-13**, onde "ctu-13" é o diretório onde se deseja salvar o arquivo com o resultado.

## Para executar o modelo com dados otimizados:

- Na linha 17 do arquivo main.py do projeto tg-botnet, faça a leitura do arquivo csv que deseja usar para alimentar o modelo.
- Na linha 18 do arquivo main.py do projeto EvoloPy-FS, informe o nome do dataset que deseja otimizar sem a extensão do arquivo, é necessário que o dataset esteja em formato csv e na pasta "datasets". Para a obtenção de resultados consistentes, é importante que seja o mesmo dataset usado no projeto tg-botnet.
- Na linha 4 do arquivo tg.sh do projeto EvoloPy-FS, indique, ao final do comando, quais algoritmos de otimização deseja usar (bat, gwo ou woa) separados por espaço. É possível usar de 1 a 3 algoritmos.
- Na linha 6 do arquivo tg.sh, informe o nome do arquivo de saída e a pasta onde ele será criado, respectivamente.

## Referências bibliográficas:

- Ruba Abu Khurma, Ibrahim Aljarah, Ahmad Sharieh, and Seyedali Mirjalili. Evolopy-fs: An

open-source nature-inspired optimization framework in python for feature selection. In Evolutionary

Machine Learning Techniques, pages 131–173. Springer, 2020

- Hossam Faris, Ibrahim Aljarah, Sayedali Mirjalili, Pedro Castillo, and J.J Merelo. "EvoloPy: An Open-source Nature-inspired Optimization Framework in Python". In Proceedings of the 8th International Joint Conference on Computational Intelligence - Volume 3: ECTA,ISBN 978-989-758-201-1, pages 171-177.

- Hossam Faris, Ali Asghar Heidari, Al-Zoubi Ala’M, Majdi Mafarja, Ibrahim Aljarah, Mohammed

Eshtay, and Seyedali Mirjalili. Time-varying hierarchical chains of salps with random weight networks

for feature selection. Expert Systems with Applications, page 112898, 2019.

- Majdi Mafarja,Ibrahim Aljarah, Ali Asghar Heidari, Hossam Faris, Philippe Fournier-Viger,Xiaodong Li, and Seyedali Mirjalili. Binary dragonfly optimization for feature selection using time-varying transfer functions. Knowledge-Based Systems, 161:185–204, 2018.

- Ibrahim Aljarah, Majdi Mafarja, Ali Asghar Heidari, Hossam Faris, Yong Zhang, and Seyedali Mirjalili. Asynchronous accelerating multi-leader salp chains for feature selection. Applied Soft Computing, 71:964–979, 2018.

- Hossam Faris, Majdi M Mafarja, Ali Asghar Heidari,Ibrahim Aljarah, Al-Zoubi Ala’M, Seyedali Mirjalili, and Hamido Fujita. An efficient binary salp swarm algorithm with crossover scheme for feature selection problems. Knowledge-Based Systems, 154:43–67, 2018.
