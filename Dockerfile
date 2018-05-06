FROM java:8u111-jdk

RUN apt-get update && apt-get install -y \
				maven \
				gradle \
                curl \
                git \
        --no-install-recommends && rm -r /var/lib/apt/lists/*

RUN cd $HOME && git clone http://github.com/davesnowdon/devoxxuk2018-dl-workshop.git
RUN cd $HOME/devoxxuk2018-dl-workshop && \
    chmod +x gradlew && \
    ./gradlew
RUN ./gradlew updateOfflineRepository -PofflineRepositoryRoot=./offline-repository
RUN ./gradlew -PofflineRepositoryRoot=./offline-repository :ex0-setup:ex0run --offline

VOLUME ./offline-repository $HOME/devoxxuk2018-dl-workshop/dl-workshop-git-repo/offline-repository
