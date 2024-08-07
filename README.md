# Timm trainer
Este repositório possui código para realizar o treinamento de modelos de classificação presentes na biblioteca Timm, assim como código para testar e fazer inferências utilizando os modelos treinados.

[Vídeo com todas as explicações]()

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

## Autor
* **Programador Artificial** - [GitHub](https://github.com/ProgramadorArtificial) - [YouTube](https://www.youtube.com/@ProgramadorArtificial)