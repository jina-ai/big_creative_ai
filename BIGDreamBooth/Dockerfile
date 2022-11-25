ARG JINA_VERSION=3.12.0


FROM jinaai/jina:${JINA_VERSION}

RUN apt-get update && apt-get install --no-install-recommends -y gcc libc6-dev git

COPY requirements.txt requirements.txt
RUN pip install --default-timeout=1000 --compile -r requirements.txt

COPY . /workdir/
WORKDIR /workdir

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]