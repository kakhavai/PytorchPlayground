# Pytorch Playground

### Installation

Install packages.

```
$ pip install -r requirements.txt
```

### Run the app

Run the app and verify cuda enabled; tensor will be random vals, cuda availability should be true.

1. python test.py

### Output:
```

VERIFY TENSOR OUTPUT:
tensor([[0.3600, 0.2383, 0.0266],
        [0.2010, 0.6393, 0.5611],
        [0.9428, 0.2870, 0.3163],
        [0.4155, 0.2017, 0.7750],
        [0.2641, 0.7674, 0.3823]])

CUDA AVAILABILITY: True
```

### Helpful misc commands -- future work

```
$ docker kill $(docker ps -q)
$ docker rm $(docker ps -a -q)
$ docker rmi $(docker images -q)
$ docker compose --env-file .env up
$ echo %CR_PAT% | docker login ghcr.io -u ${{ github.actor }} --password-stdin 
$ docker pull ghcr.io/kakhavai/kian-wiki-database
$ docker pull ghcr.io/kakhavai/kian-wiki-nginx
$ docker pull ghcr.io/kakhavai/kian-wiki-nodeserver
$ POSTGRES_HOST=localhost
$ docker-compose -f docker-compose-local.yml up
$ wsl --shutdown
```
