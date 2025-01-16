# Timm trainer
Este repositório possui código para realizar o treinamento de modelos de classificação e embedding presentes na biblioteca Timm, assim como código para testar e fazer inferências utilizando os modelos treinados.

[Vídeo com todas as explicações_classificação](https://www.youtube.com/watch?v=WBWh7FINHQU)

[Vídeo com todas as explicações_embedding]()

## Pré-requisitos
Não existem requisitos específicos, mas segue abaixo as versões das principais bibliotecas utilizadas:

- Python == 3.12.3
- Timm == 1.0.7
- Torch == 2.3.0
- Torchvision == 0.18.0 

## Organização do dataset
O dataset deve estar no seguinte formato:

```bash
dataset
├── nome_do_dataset
│   ├── classe1
│   │   ├── imagem1.jpg
│   │   ├── imagem2.jpg
│   ├── classe2
│   │   ├── imagem1.jpg
```

Sendo que o "nome_do_dataset" pode ser qualquer um, assim como o nome das classes e o nome das imagens. Atentar-se apenas que o nome das pastas das classes serão utilizadas para ajudar nos testes e nas inferências após realizar o treinamento do modelo.

## Como rodar
Todas as configurações estão no próprio código e nos respectivos vídeos!

Boa parte do código é compartilhado entre os modelos de classificação e embedding, com exceção dos códigos de treinamento, testes e inferências devido a customizações específicas para cada um deles. Para facilitar a separação deles foi adicionado no nome dos arquivos a sigla "cla" para códigos referentes a arquiteturas de classificação e "emb" para códigos referentes a arquiteturas de embedding. Assim como uma numeração indicando a ordem que os códigos serão normalmente executados.

## Autor
* **Programador Artificial** - [GitHub](https://github.com/ProgramadorArtificial) - [YouTube](https://www.youtube.com/@ProgramadorArtificial)
