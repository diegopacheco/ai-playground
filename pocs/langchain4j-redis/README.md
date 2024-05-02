### Build 
```bash
./mvnw clean install 
```
### Result
* LangChain
* use Redis as vector database for embeddings
* use Redis to find relevance between different sentences(embeddings)
```
[Main.main()] INFO org.testcontainers.images.PullPolicy - Image pull policy will be performed by: DefaultPullPolicy()
[Main.main()] INFO org.testcontainers.utility.ImageNameSubstitutor - Image name substitution will be performed by: DefaultImageNameSubstitutor (composite of 'ConfigurationFileImageNameSubstitutor' and 'PrefixingImageNameSubstitutor')
[Main.main()] INFO org.testcontainers.dockerclient.DockerClientProviderStrategy - Loaded org.testcontainers.dockerclient.UnixSocketClientProviderStrategy from ~/.testcontainers.properties, will try it first
[Main.main()] INFO org.testcontainers.dockerclient.DockerClientProviderStrategy - Found Docker environment with local Unix socket (unix:///var/run/docker.sock)
[Main.main()] INFO org.testcontainers.DockerClientFactory - Docker host IP address is localhost
[Main.main()] INFO org.testcontainers.DockerClientFactory - Connected to docker:
  Server Version: 19.03.11
  API Version: 1.40
  Operating System: Ubuntu 22.04.4 LTS
  Total Memory: 64150 MB
[Main.main()] WARN org.testcontainers.utility.ResourceReaper -
********************************************************************************
Ryuk has been disabled. This can cause unexpected behavior in your environment.
********************************************************************************
[Main.main()] INFO org.testcontainers.DockerClientFactory - Checking the system...
[Main.main()] INFO org.testcontainers.DockerClientFactory - ✔︎ Docker server version should be at least 1.6.0
[Main.main()] INFO tc.redis/redis-stack-server:latest - Creating container for image: redis/redis-stack-server:latest
[Main.main()] INFO tc.redis/redis-stack-server:latest - Container redis/redis-stack-server:latest is starting: 34e158d2b4755ccae83c21897adcbfb552c65518c5ab4704a87093afd6b2b84a
[Main.main()] INFO tc.redis/redis-stack-server:latest - Container redis/redis-stack-server:latest started in PT0.671887581S
[Main.main()] INFO ai.djl.util.Platform - Found matching platform from: jar:file:/home/diego/.m2/repository/ai/djl/huggingface/tokenizers/0.26.0/tokenizers-0.26.0.jar!/native/lib/tokenizers.properties
0.8144288659095
I like football.
```