## Now

- [x] format documents returned by database correctly
- [x] remove duplicates in results by id and chunk index
- get citations working again
  - figure out inline citations
- [x] get tools to work
- [x] just do requests, no ollama client library
- Rework crawler
  - They should import crawler and set extractor and then call run in their process_my_files.py file.
- [x] Make the potential loop with MAX_TOOL_CALLS work
- [x] Simplify event emitter code
- Add user valves where they can put in username and password for database token
- enable streaming again for final response
- tech talk on 15, must be ready by then.

- Crawler

  - logging
  - Demo with 2 sources of documents

- Radchat

  - [x] Implement RAG with function call to get filters and queries
    - [x] Check what metadata should be returned
    - Find a way to get the metadata schema from the collection so it can properly filter on metadata
  - [ ] Clean up function
    - [ ] Remove Groq
    - [ ] Use langchain? and .with_structured_output
    - Revoke groq API key, it is exposed.
  - [ ] Use with bigger document collection
  - [ ] Implement citations
  - [ ] Implement event_emitter

- Implement security features for milvus
  - Add them to config files
  - Set up 2 repositories with different types of data
  - Test them based on user details in python script (not necessarily in ui)
  - upload different directory too
- Using basic_ollama as function in OI.
- Standardize configuration

- use cloudflared tunnel --url http://localhost:<desired-port> to expose the service to the internet
- maybe can do ingresses exposing http://ollama.localhost:5000 and similarly for the other services.

https://github.com/PatrickJS/awesome-cursorrules/blob/main/rules/htmx-go-basic-cursorrules-prompt-file/.cursorrules

# Crawler System Improvement TODOs

- [ ] Configuration (`config`)

  - [ ] Rename `collection_template.yaml` to something clearer like `default_collection_config.yaml`.
  - [ ] Restructure directory-specific config (`directories/*.yaml`) for better clarity (e.g., explicit `target_collection`, `collection_overrides`, `processing` sections).
  - [ ] Change `ConfigManager` to load and key directory configs by _name_ (e.g., "conference") instead of the `path` value within the file.
  - [ ] Implement configuration validation (e.g., using Pydantic or JSON Schema) to check for required fields and correct types.
  - [ ] Replace `print` with `logging` for error reporting in `ConfigManager`.
  - [ ] Ensure extractor configuration (`processing.extractors`) in the effective config correctly enables/disables specific readers in the `Extractor`.

- [ ] Architecture & Core Components (`architecture`)

  - [ ] Refactor `Extractor` to separate responsibilities: Coordinate extraction, select Reader based on config, delegate LLM metadata extraction to a separate step/component.
  - [ ] Modify `Extractor` to only initialize document readers (`self.doc_readers`) that are enabled in the effective configuration for the current directory.
  - [ ] Review `DocumentContent`: Separate raw text blocks from descriptive placeholders for images/tables to avoid duplication in `get_text()`.
  - [ ] Ensure `Crawler`, `DocumentProcessor`, `Extractor`, `Embedder`, and `Storage` classes maintain clear and distinct responsibilities after refactoring.

- [ ] Data Flow & Processing (`processing`)

  - [ ] **Critical:** Fix memory issue in `main.py` by implementing streaming/batch processing and insertion into `VectorStorage` instead of collecting all results first.
  - [ ] **Critical:** Implement document chunking logic (e.g., add a `Chunker` component).
  - [ ] Modify the processing pipeline (`DocumentProcessor` or `Crawler`) to iterate over chunks instead of whole documents.
  - [ ] Adapt embedding generation to work on a per-chunk basis.
  - [ ] Design and implement strategy for handling metadata with chunking (document-level vs. chunk-level metadata).
  - [ ] Implement logic to handle `metadata.extra_embeddings` configuration for embedding specific metadata fields.
  - [ ] Enhance file discovery (`_setup_filepaths`) to filter by configured extensions and potentially add exclusion patterns.
  - [ ] Decide and implement strategy for LLM metadata extraction in the context of chunking (once per doc or per chunk).

- [ ] Storage (`storage`)

  - [ ] Fix `VectorStorage.__enter__` to load an existing Milvus collection instead of dropping and recreating it. Dropping should be a separate, explicit operation.
  - [ ] Refine `build_collection_schema` in `vector_db.py`:
    - [ ] Ensure fields (`id`, `embedding`, `text`) are defined consistently and only once, respecting the schema config.
    - [ ] Align `maxLength` definitions for `VARCHAR` fields between the schema config and Milvus schema generation.
    - [ ] Document the JSON `array` to Milvus `VARCHAR` mapping or investigate native `ARRAY` type.
  - [ ] Align `MAX_DOC_LENGTH` constant in `vector_db.py` with schema definitions and actual Milvus capabilities/limits derived from config.
  - [ ] Re-evaluate and potentially implement an efficient duplicate checking mechanism in `insert_data` if needed (consider Milvus UPSERT).
  - [ ] Add robust error handling and logging for Milvus connection, insertion, and search operations.
  - [ ] Update `VectorStorage` schema/insertion logic to correctly store document-level and chunk-level metadata.

- [ ] Code Quality (`quality`)

  - [ ] Replace all `print` statements used for logging/status/errors with the standard `logging` module.
  - [ ] Implement comprehensive `try-except` blocks around I/O, network calls, and parsing. Log errors effectively.
  - [ ] Ensure consistent naming conventions (e.g., resolve `directory_name` vs. `dir_path`).
  - [ ] Standardize override structures in configuration files.
  - [ ] Refactor large methods into smaller, focused functions for readability.
  - [ ] Ensure all dependencies (including optional ones for readers) are listed correctly in `requirements.txt` (consider `extras_require`).
  - [ ] Review and update docstrings and comments to reflect code changes after refactoring.

- [ ] Examples & Documentation (`docs`)
  - [ ] Update `examples/process_documents.py` to reflect the refactored architecture and demonstrate key features (config, batching, chunking).
  - [ ] Update `README.md` and `Crawler.md` to accurately describe the final architecture, configuration, and usage.
  - [ ] Update `processing/processing.md` based on the refactored structure.
  - [ ] Update `storage/db.md` to reflect the final `VectorStorage` implementation and configuration.

# MVP

## General

- [ ] Data sources

  - [ ] What local data do I download
    - [ ] General conference talks
    - [ ] Scriptures in chapters
  - [ ] What search apis do I use
    - [ ] brave search
  - [ ] What crawling do I do?
    - [ ] Levels deep of the link graph
    - [ ] How many links to follow
  - [ ] How to manage citations

- [x] Make repo public
- If using openwebui, include instructions for it's deployment and special config
  - Include pipelines

## Frontend

## Backend

- [x] Decide framework
  - Python then after MVP -> Go
- [ ] Create good async enabled load balancer

## Future

- [ ] After MVP
  - [ ] Refactor backend into Go
  - [ ] Create the small model fine tuning and test time inference, then run locally

Okay, let's go through this systematically. I'll compare your `values.yaml` settings with the rendered templates for each component.

**General Observations:**

- **Release Name & Namespace:** Your release name `ip` and namespace `test` are consistently applied across resources. This is good.
- **Umbrella Chart Structure:** You're using an umbrella chart, so values for subcharts (milvus, open-webui, ollama) are nested under their respective keys (e.g., `milvus.persistence.size`). This is the correct approach.

Let's break it down by component:

**1. Milvus & its Subcharts (Minio, Etcd)**

- **Milvus itself:**

  - `image.all.tag: "v2.4.15"`
    - **TEMPLATE CHECK:** `ip-milvus-standalone` Deployment uses `image: "milvusdb/milvus:v2.4.15"`.
    - **STATUS: CORRECTLY APPLIED.**
  - `persistence.size: 16Gi`
    - **TEMPLATE CHECK:** `ip-milvus` PVC requests `storage: 50Gi`.
    - **STATUS: MISMATCH.** Your value of `16Gi` is not being applied. The Milvus chart is likely using a default or has a different specific key for standalone persistence size.
    - **Suggestion:** Check the Milvus subchart's `values.yaml` (you can get it via `helm show values zilliztech/milvus`). It might be something like `standalone.persistence.size` or similar. The path `milvus.persistence.size` might apply to a different component or mode within the Milvus chart.
  - `standalone.resources`:
    - Requests: `cpu: 1000m`, `memory: 2Gi`
    - Limits: `cpu: 2000m`, `memory: 4Gi`
    - **TEMPLATE CHECK:** `ip-milvus-standalone` Deployment has these exact resources.
    - **STATUS: CORRECTLY APPLIED.**

- **Minio (subchart of Milvus):**

  - `milvus.minio.persistence.size: 16Gi`
    - **TEMPLATE CHECK:** `ip-minio` PVC requests `storage: "16Gi"`.
    - **STATUS: CORRECTLY APPLIED.**
  - `milvus.minio.resources`:
    - Requests: `cpu: 250m`, `memory: 512Mi`
    - Limits: `cpu: 500m`, `memory: 1Gi`
    - **TEMPLATE CHECK:** `ip-minio` Deployment has these exact resources.
    - **STATUS: CORRECTLY APPLIED.**
  - `milvus.minio.image`: You haven't specified an image for Minio in your values, so it's using the default from the Minio subchart: `minio/minio:RELEASE.2023-03-20T20-16-18Z`. This is expected.

- **ETCD (subchart of Milvus):**
  - `milvus.etcd.replicaCount: 1`
    - **TEMPLATE CHECK:** `ip-etcd` StatefulSet has `replicas: 1`.
    - **STATUS: CORRECTLY APPLIED.**
  - Other ETCD settings (image, persistence) are using defaults from the ETCD subchart, as you haven't overridden them. This is expected. For example, ETCD PVC requests `10Gi`.

**2. OpenWebUI & its Subchart (Pipelines)**

- **OpenWebUI itself:**

  - `image.tag: "main"`
    - **TEMPLATE CHECK:** `open-webui` StatefulSet uses `image: ghcr.io/open-webui/open-webui:main`.
    - **STATUS: CORRECTLY APPLIED.**
  - `persistence.size: 8Gi`
    - **TEMPLATE CHECK:** `open-webui` PVC requests `storage: 8Gi`.
    - **STATUS: CORRECTLY APPLIED.**
  - `resources`:
    - Requests: `cpu: 500m`, `memory: 1Gi`
    - Limits: `cpu: 1000m`, `memory: 2Gi`
    - **TEMPLATE CHECK:** `open-webui` StatefulSet container has these exact resources.
    - **STATUS: CORRECTLY APPLIED.**
  - `ingress.enabled: true` (implied by Ingress resource existing)
  - `ingress.class: "nginx"`
    - **TEMPLATE CHECK:** `open-webui` Ingress has `ingressClassName: nginx`.
    - **STATUS: CORRECTLY APPLIED.**
  - `ingress.host: "oi.{{ .Values.global.domain }}"` with `global.domain: api.meatheadmathematician.com`
    - **TEMPLATE CHECK:** `open-webui` Ingress has `host: oi.api.meatheadmathematician.com`.
    - **STATUS: CORRECTLY APPLIED.** The templating with `global.domain` worked.
  - `ollama.enabled: false`
    - **TEMPLATE CHECK:** `open-webui` StatefulSet container env has `ENABLE_OLLAMA_API: "False"`.
    - **STATUS: CORRECTLY APPLIED.**
  - `ollamaurls: - "http://ollama.{{ .Values.global.environment }}.svc.cluster.local:11434"`
    - **TEMPLATE CHECK:** `open-webui` StatefulSet container env has:
      `OPENAI_API_BASE_URLS: "http://open-webui-pipelines.test.svc.cluster.local:9099;https://api.openai.com/v1"`
    - **STATUS: MISMATCH / NOT USED AS INTENDED.** Your `ollamaurls` key is not being used to populate `OPENAI_API_BASE_URLS`. The OpenWebUI chart seems to be constructing this URL from its `pipelines` subchart and a default OpenAI URL. The `ollamaurls` key you've defined is likely a custom key that the OpenWebUI subchart doesn't recognize for this purpose.
    - **Suggestion:** You need to find out how the OpenWebUI chart expects external Ollama URLs to be configured. Check its `values.yaml` (e.g., `helm show values open-webui/open-webui`). It might be a specific key like `config.ollamaBaseUrl` or you might need to set `OPENAI_API_BASE_URLS` directly under `open-webui.extraEnvVars` or `open-webui.config` or a similar section if the chart supports it. Your current `ollamaurls` key is effectively an unused custom value at the `open-webui` level in your umbrella chart.

- **Pipelines (subchart of OpenWebUI):**
  - You haven't specified any values for `open-webui.pipelines`, so it's using its defaults (e.g., PVC size `2Gi`, image `ghcr.io/open-webui/pipelines:main`). This is expected.

**3. Ollama**

- `image.tag: "latest"`
  - **TEMPLATE CHECK:** `ip-ollama` Deployment uses `image: "ollama/ollama:latest"`.
  - **STATUS: CORRECTLY APPLIED.**
- `service.port: 11434`
  - **TEMPLATE CHECK:** `ip-ollama` Service has `port: 11434`.
  - **STATUS: CORRECTLY APPLIED.**
- `persistence.enabled: true` and `persistence.size: 24Gi`
  - **TEMPLATE CHECK:** `ip-ollama` Deployment defines a volume `ollama-data` as `emptyDir: {}`. No PVC is created for Ollama.
  - **STATUS: MISMATCH.** Your persistence settings for Ollama are not being applied. The Ollama chart is using an `emptyDir` instead of a PVC.
  - **Suggestion:** Check the Ollama subchart's `values.yaml` (e.g., `helm show values otwld/ollama`). The way to enable PVCs might be different. Common patterns include:
    - A top-level `persistence.enabled: true` within the `ollama` section of _its own values_, and then sub-keys for `size` and `storageClass`.
    - A specific key like `ollama.persistence.type: pvc` or `ollama.volume.type: pvc`.
    - The subchart might look for `ollama.persistence.existingClaim` and if not found and `ollama.persistence.enabled` is true, it creates one. Your current structure `ollama.persistence.size` might not be recognized by the subchart directly when nested. Ensure the path is exactly what the subchart expects.
- `resources`:
  - Requests: `cpu: 2000m`, `memory: 4Gi`
  - Limits: `cpu: 4000m`, `memory: 8Gi`
  - **TEMPLATE CHECK:** `ip-ollama` Deployment container has these exact resources.
  - **STATUS: CORRECTLY APPLIED.**
- `models.pull`:
  - `pull: - qwen3:1.7b ...`
  - **TEMPLATE CHECK:** There's no initContainer or Job in the provided Ollama templates that would use this list to pull models.
  - **STATUS: NOT USED / FEATURE MISMATCH.** The `ollama` Helm chart you're using (`otwld/ollama-helm`) might not support this `models.pull` feature directly in its templates, or it expects it under a different key.
  - **Suggestion:** Check the documentation or `values.yaml` of the `otwld/ollama-helm` chart to see if and how it supports pre-pulling models. It might require a specific job to be enabled or different value keys.

**Summary of Key Issues & Recommendations:**

1.  **Milvus Standalone PVC Size:**

    - **Issue:** Your `milvus.persistence.size: 16Gi` is not respected; template shows `50Gi`.
    - **Action:** Check the Milvus chart's `values.yaml` for the correct key to set standalone PVC size. It might be `milvus.standalone.persistence.size` or similar.

2.  **OpenWebUI Ollama URL Configuration:**

    - **Issue:** Your `open-webui.ollamaurls` key is not used. The `OPENAI_API_BASE_URLS` env var is set by OpenWebUI's internal logic.
    - **Action:** Consult the OpenWebUI chart's `values.yaml`. You likely need to use a specific key provided by that chart to configure external Ollama URLs, or directly set the `OPENAI_API_BASE_URLS` environment variable if the chart allows overriding it (e.g., via `open-webui.env` or `open-webui.extraEnvVars`).

3.  **Ollama Persistence:**

    - **Issue:** Your `ollama.persistence.enabled: true` and `ollama.persistence.size: 24Gi` are ignored; template uses `emptyDir`.
    - **Action:** Examine the Ollama chart's `values.yaml`. The subchart has its own way of enabling and configuring PVCs. You need to match its expected key structure (e.g., it might be `ollama.persistence.enabled` and `ollama.persistence.size` _within the context of the subchart's values_, or perhaps `ollama.volumes.ollama_data.persistentVolumeClaim.create: true` and `ollama.volumes.ollama_data.persistentVolumeClaim.size: 24Gi`).

4.  **Ollama Model Pre-pulling:**
    - **Issue:** The `ollama.models.pull` value doesn't seem to be used by any generated resources.
    - **Action:** Verify if the `otwld/ollama-helm` chart version you're using supports this feature and, if so, under what key.

**General Advice for Debugging Umbrella Charts:**

- **`helm show values <repository>/<chartname>`:** Always start by looking at the `values.yaml` of the subchart itself to understand the available configuration options and their exact paths. For example:
  - `helm show values zilliztech/milvus`
  - `helm show values open-webui/open-webui`
  - `helm show values otwld/ollama`
- **Path Specificity:** When setting values for a subchart in an umbrella chart, the path must exactly match what the subchart expects, prefixed by the subchart's alias in your `Chart.yaml` (e.g., `milvus.`, `open-webui.`, `ollama.`).
- **`helm template --debug <releasename> . -f values.yaml`:** The `--debug` flag can sometimes give more insight into how values are being merged and interpreted, though it can be verbose.

You're on the right track, and most of your values are being applied correctly! The mismatches are common when working with umbrella charts and subchart configurations. Just a bit more digging into the subcharts' specific value keys should resolve these.

---

# Source: internal-perplexity/charts/milvus/charts/minio/templates/serviceaccount.yaml

apiVersion: v1
kind: ServiceAccount
metadata:
name: "ip-minio"
namespace: "test"
labels:
app: minio
chart: minio-8.0.17
release: "ip"

---

# Source: internal-perplexity/charts/ollama/templates/serviceaccount.yaml

apiVersion: v1
kind: ServiceAccount
metadata:
name: ip-ollama
namespace: test
labels:
helm.sh/chart: ollama-1.18.0
app.kubernetes.io/name: ollama
app.kubernetes.io/instance: ip
app.kubernetes.io/version: "0.7.1"
app.kubernetes.io/managed-by: Helm
automountServiceAccountToken: true

---

# Source: internal-perplexity/charts/open-webui/charts/pipelines/templates/service-account.yaml

apiVersion: v1
kind: ServiceAccount
metadata:
name: open-webui-pipelines
namespace: test
labels:
helm.sh/chart: pipelines-0.7.0
app.kubernetes.io/version: "alpha"
app.kubernetes.io/managed-by: Helm
app.kubernetes.io/instance: ip
app.kubernetes.io/component: open-webui-pipelines

---

# Source: internal-perplexity/charts/open-webui/templates/service-account.yaml

apiVersion: v1
kind: ServiceAccount
metadata:
name: open-webui
namespace: test
labels:
helm.sh/chart: open-webui-6.17.0
app.kubernetes.io/version: "0.6.11"
app.kubernetes.io/managed-by: Helm
app.kubernetes.io/instance: ip
app.kubernetes.io/component: open-webui
automountServiceAccountToken: false

---

# Source: internal-perplexity/charts/milvus/charts/minio/templates/secrets.yaml

apiVersion: v1
kind: Secret
metadata:
name: ip-minio
labels:
app: minio
chart: minio-8.0.17
release: ip
heritage: Helm
type: Opaque
data:
accesskey: "bWluaW9hZG1pbg=="
secretkey: "bWluaW9hZG1pbg=="

---

# Source: internal-perplexity/charts/milvus/charts/minio/templates/configmap.yaml

apiVersion: v1
kind: ConfigMap
metadata:
name: ip-minio
labels:
app: minio
chart: minio-8.0.17
release: ip
heritage: Helm
data:
initialize: |-
#!/bin/sh
set -e ; # Have script exit in the event of a failed command.
MC_CONFIG_DIR="/etc/minio/mc/"
MC="/usr/bin/mc --insecure --config-dir ${MC_CONFIG_DIR}"

    # connectToMinio
    # Use a check-sleep-check loop to wait for Minio service to be available
    connectToMinio() {
      SCHEME=$1
      ATTEMPTS=0 ; LIMIT=29 ; # Allow 30 attempts
      set -e ; # fail if we can't read the keys.
      ACCESS=$(cat /config/accesskey) ; SECRET=$(cat /config/secretkey) ;
      set +e ; # The connections to minio are allowed to fail.
      echo "Connecting to Minio server: $SCHEME://$MINIO_ENDPOINT:$MINIO_PORT" ;
      MC_COMMAND="${MC} config host add myminio $SCHEME://$MINIO_ENDPOINT:$MINIO_PORT $ACCESS $SECRET" ;
      $MC_COMMAND ;
      STATUS=$? ;
      until [ $STATUS = 0 ]
      do
        ATTEMPTS=`expr $ATTEMPTS + 1` ;
        echo \"Failed attempts: $ATTEMPTS\" ;
        if [ $ATTEMPTS -gt $LIMIT ]; then
          exit 1 ;
        fi ;
        sleep 2 ; # 1 second intervals between attempts
        $MC_COMMAND ;
        STATUS=$? ;
      done ;
      set -e ; # reset `e` as active
      return 0
    }

    # checkBucketExists ($bucket)
    # Check if the bucket exists, by using the exit code of `mc ls`
    checkBucketExists() {
      BUCKET=$1
      CMD=$(${MC} ls myminio/$BUCKET > /dev/null 2>&1)
      return $?
    }

    # createBucket ($bucket, $policy, $purge)
    # Ensure bucket exists, purging if asked to
    createBucket() {
      BUCKET=$1
      POLICY=$2
      PURGE=$3
      VERSIONING=$4

      # Purge the bucket, if set & exists
      # Since PURGE is user input, check explicitly for `true`
      if [ $PURGE = true ]; then
        if checkBucketExists $BUCKET ; then
          echo "Purging bucket '$BUCKET'."
          set +e ; # don't exit if this fails
          ${MC} rm -r --force myminio/$BUCKET
          set -e ; # reset `e` as active
        else
          echo "Bucket '$BUCKET' does not exist, skipping purge."
        fi
      fi

      # Create the bucket if it does not exist
      if ! checkBucketExists $BUCKET ; then
        echo "Creating bucket '$BUCKET'"
        ${MC} mb myminio/$BUCKET
      else
        echo "Bucket '$BUCKET' already exists."
      fi


      # set versioning for bucket
      if [ ! -z $VERSIONING ] ; then
        if [ $VERSIONING = true ] ; then
            echo "Enabling versioning for '$BUCKET'"
            ${MC} version enable myminio/$BUCKET
        elif [ $VERSIONING = false ] ; then
            echo "Suspending versioning for '$BUCKET'"
            ${MC} version suspend myminio/$BUCKET
        fi
      else
          echo "Bucket '$BUCKET' versioning unchanged."
      fi

      # At this point, the bucket should exist, skip checking for existence
      # Set policy on the bucket
      echo "Setting policy of bucket '$BUCKET' to '$POLICY'."
      ${MC} policy set $POLICY myminio/$BUCKET
    }

    # Try connecting to Minio instance
    scheme=http
    connectToMinio $scheme

---

# Source: internal-perplexity/charts/milvus/templates/configmap.yaml

# If customConfigMap is not set, this ConfigMap will be redendered.

apiVersion: v1
kind: ConfigMap
metadata:
name: ip-milvus
namespace: test
data:
default.yaml: |+ # Copyright (C) 2019-2021 Zilliz. All rights reserved. # # Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance # with the License. You may obtain a copy of the License at # # http://www.apache.org/licenses/LICENSE-2.0 # # Unless required by applicable law or agreed to in writing, software distributed under the License # is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express # or implied. See the License for the specific language governing permissions and limitations under the License.

    etcd:
      endpoints:
      - ip-etcd-0.ip-etcd-headless.test.svc.cluster.local:2379

    metastore:
      type: etcd

    minio:
      address: ip-minio
      port: 9000
      accessKeyID: minioadmin
      secretAccessKey: minioadmin
      useSSL: false
      bucketName: milvus-bucket
      rootPath: file
      useIAM: false
      useVirtualHost: false

    mq:
      type: rocksmq

    messageQueue: rocksmq

    rootCoord:
      address: localhost
      port: 53100
      enableActiveStandby: false  # Enable rootcoord active-standby

    proxy:
      port: 19530
      internalPort: 19529

    queryCoord:
      address: localhost
      port: 19531

      enableActiveStandby: false  # Enable querycoord active-standby

    queryNode:
      port: 21123
      enableDisk: true # Enable querynode load disk index, and search on disk index

    indexCoord:
      address: localhost
      port: 31000
      enableActiveStandby: false  # Enable indexcoord active-standby

    indexNode:
      port: 21121
      enableDisk: true # Enable index node build disk vector index

    dataCoord:
      address: localhost
      port: 13333
      enableActiveStandby: false  # Enable datacoord active-standby

    dataNode:
      port: 21124

    log:
      level: info
      file:
        rootPath: ""
        maxSize: 300
        maxAge: 10
        maxBackups: 20
      format: text

user.yaml: |- # For example enable rest http for milvus proxy # proxy: # http: # enabled: true # maxUserNum: 100 # maxRoleNum: 10 ## Enable tlsMode and set the tls cert and key # tls: # serverPemPath: /etc/milvus/certs/tls.crt # serverKeyPath: /etc/milvus/certs/tls.key # common: # security: # tlsMode: 1

---

# Source: internal-perplexity/charts/milvus/charts/minio/templates/pvc.yaml

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
name: ip-minio
annotations:
helm.sh/resource-policy: keep
labels:
app: minio
chart: minio-8.0.17
release: ip
heritage: Helm
spec:
accessModes: - "ReadWriteOnce"
resources:
requests:
storage: "16Gi"

---

# Source: internal-perplexity/charts/milvus/templates/pvc.yaml

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
name: ip-milvus
namespace: test
annotations:
helm.sh/resource-policy: keep
labels:
helm.sh/chart: milvus-4.2.49
app.kubernetes.io/name: milvus
app.kubernetes.io/instance: ip
app.kubernetes.io/version: "2.5.12"
app.kubernetes.io/managed-by: Helm
spec:
accessModes:

- "ReadWriteOnce"
  resources:
  requests:
  storage: 50Gi

---

# Source: internal-perplexity/charts/open-webui/charts/pipelines/templates/pvc.yaml

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
name: open-webui-pipelines
namespace: test
labels:
app.kubernetes.io/instance: ip
app.kubernetes.io/component: open-webui-pipelines
spec:
accessModes: - "ReadWriteOnce"
resources:
requests:
storage: 2Gi

---

# Source: internal-perplexity/charts/open-webui/templates/pvc.yaml

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
name: open-webui
namespace: test
labels:
app.kubernetes.io/instance: ip
app.kubernetes.io/component: open-webui
spec:
accessModes: - "ReadWriteOnce"
resources:
requests:
storage: 8Gi

---

# Source: internal-perplexity/charts/milvus/charts/etcd/templates/svc-headless.yaml

apiVersion: v1
kind: Service
metadata:
name: ip-etcd-headless
namespace: test
labels:
app.kubernetes.io/name: etcd
helm.sh/chart: etcd-6.3.3
app.kubernetes.io/instance: ip
app.kubernetes.io/managed-by: Helm
annotations:
service.alpha.kubernetes.io/tolerate-unready-endpoints: "true"
spec:
type: ClusterIP
clusterIP: None
publishNotReadyAddresses: true
ports: - name: "client"
port: 2379
targetPort: client - name: "peer"
port: 2380
targetPort: peer
selector:
app.kubernetes.io/name: etcd
app.kubernetes.io/instance: ip

---

# Source: internal-perplexity/charts/milvus/charts/etcd/templates/svc.yaml

apiVersion: v1
kind: Service
metadata:
name: ip-etcd
namespace: test
labels:
app.kubernetes.io/name: etcd
helm.sh/chart: etcd-6.3.3
app.kubernetes.io/instance: ip
app.kubernetes.io/managed-by: Helm
annotations:
spec:
type: ClusterIP
ports: - name: "client"
port: 2379
targetPort: client
nodePort: null - name: "peer"
port: 2380
targetPort: peer
nodePort: null
selector:
app.kubernetes.io/name: etcd
app.kubernetes.io/instance: ip

---

# Source: internal-perplexity/charts/milvus/charts/minio/templates/service.yaml

apiVersion: v1
kind: Service
metadata:
name: ip-minio
labels:
app: minio
chart: minio-8.0.17
release: ip
heritage: Helm
spec:
type: ClusterIP
ports: - name: http
port: 9000
protocol: TCP
targetPort: 9000
selector:
app: minio
release: ip

---

# Source: internal-perplexity/charts/milvus/templates/service.yaml

apiVersion: v1
kind: Service
metadata:
name: ip-milvus
namespace: test
labels:
helm.sh/chart: milvus-4.2.49
app.kubernetes.io/name: milvus
app.kubernetes.io/instance: ip
app.kubernetes.io/version: "2.5.12"
app.kubernetes.io/managed-by: Helm
component: "standalone"
spec:
type: ClusterIP
ports: - name: milvus
port: 19530
protocol: TCP
targetPort: milvus - name: metrics
protocol: TCP
port: 9091
targetPort: metrics
selector:
app.kubernetes.io/name: milvus
app.kubernetes.io/instance: ip
component: "standalone"

---

# Source: internal-perplexity/charts/ollama/templates/service.yaml

apiVersion: v1
kind: Service
metadata:
name: ip-ollama
namespace: test
labels:
helm.sh/chart: ollama-1.18.0
app.kubernetes.io/name: ollama
app.kubernetes.io/instance: ip
app.kubernetes.io/version: "0.7.1"
app.kubernetes.io/managed-by: Helm
spec:
type: ClusterIP
ports: - port: 11434
targetPort: http
protocol: TCP
name: http
selector:
app.kubernetes.io/name: ollama
app.kubernetes.io/instance: ip

---

# Source: internal-perplexity/charts/open-webui/charts/pipelines/templates/service.yaml

apiVersion: v1
kind: Service
metadata:
name: open-webui-pipelines
namespace: test
labels:
helm.sh/chart: pipelines-0.7.0
app.kubernetes.io/version: "alpha"
app.kubernetes.io/managed-by: Helm
app.kubernetes.io/instance: ip
app.kubernetes.io/component: open-webui-pipelines
spec:
selector:
app.kubernetes.io/instance: ip
app.kubernetes.io/component: open-webui-pipelines
type: ClusterIP
ports:

- protocol: TCP
  name: http
  port: 9099
  targetPort: http

---

# Source: internal-perplexity/charts/open-webui/templates/service.yaml

apiVersion: v1
kind: Service
metadata:
name: open-webui
namespace: test
labels:
helm.sh/chart: open-webui-6.17.0
app.kubernetes.io/version: "0.6.11"
app.kubernetes.io/managed-by: Helm
app.kubernetes.io/instance: ip
app.kubernetes.io/component: open-webui
spec:
selector:
app.kubernetes.io/instance: ip
app.kubernetes.io/component: open-webui
type: ClusterIP
ports:

- protocol: TCP
  name: http
  port: 8080
  targetPort: http

---

# Source: internal-perplexity/charts/milvus/charts/minio/templates/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
name: ip-minio
labels:
app: minio
chart: minio-8.0.17
release: ip
heritage: Helm
spec:
strategy:
type: RollingUpdate
rollingUpdate:
maxSurge: 100%
maxUnavailable: 0
selector:
matchLabels:
app: minio
release: ip
template:
metadata:
name: ip-minio
labels:
app: minio
release: ip
annotations:
checksum/secrets: e5b865008ba6252597a066f4a2d4d29f8133539fdb711ceedabc0ea2235cd89c
checksum/config: c5c2bdb6403baebdb4f10d4848e8c0231c4484868bf53ba81eccabaf54b3ed9a
spec:
serviceAccountName: "ip-minio"
securityContext:
runAsUser: 1000
runAsGroup: 1000
fsGroup: 1000
containers: - name: minio
image: "minio/minio:RELEASE.2023-03-20T20-16-18Z"
imagePullPolicy: IfNotPresent
command: - "/bin/sh" - "-ce" - "/usr/bin/docker-entrypoint.sh minio -S /etc/minio/certs/ server /export"
volumeMounts: - name: export
mountPath: /export  
 ports: - name: http
containerPort: 9000
livenessProbe:
httpGet:
path: /minio/health/live
port: http
scheme: HTTP
initialDelaySeconds: 5
periodSeconds: 5
timeoutSeconds: 5
successThreshold: 1
failureThreshold: 5
readinessProbe:
tcpSocket:
port: http
initialDelaySeconds: 5
periodSeconds: 5
timeoutSeconds: 1
successThreshold: 1
failureThreshold: 5
startupProbe:
tcpSocket:
port: http
initialDelaySeconds: 0
periodSeconds: 10
timeoutSeconds: 5
successThreshold: 1
failureThreshold: 60
env: - name: MINIO_ACCESS_KEY
valueFrom:
secretKeyRef:
name: ip-minio
key: accesskey - name: MINIO_SECRET_KEY
valueFrom:
secretKeyRef:
name: ip-minio
key: secretkey
resources:
limits:
cpu: 500m
memory: 1Gi
requests:
cpu: 250m
memory: 512Mi  
 volumes: - name: export
persistentVolumeClaim:
claimName: ip-minio - name: minio-user
secret:
secretName: ip-minio

---

# Source: internal-perplexity/charts/milvus/templates/standalone-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
name: ip-milvus-standalone
namespace: test
labels:
helm.sh/chart: milvus-4.2.49
app.kubernetes.io/name: milvus
app.kubernetes.io/instance: ip
app.kubernetes.io/version: "2.5.12"
app.kubernetes.io/managed-by: Helm
component: "standalone"

annotations:

spec:
replicas: 1
strategy:
type: Recreate
selector:
matchLabels:
app.kubernetes.io/name: milvus
app.kubernetes.io/instance: ip
component: "standalone"
template:
metadata:
labels:
app.kubernetes.io/name: milvus
app.kubernetes.io/instance: ip
component: "standalone"

      annotations:
        checksum/config: f94eb3ce806bf17e15a71e2c32acc00c06c3120bd21bdcbdd90f7d710bd122d2

    spec:
      serviceAccountName: default
      initContainers:
      containers:
      - name: standalone
        image: "milvusdb/milvus:v2.4.15"
        imagePullPolicy: Always
        args: [ "milvus", "run", "standalone" ]
        ports:
          - name: milvus
            containerPort: 19530
            protocol: TCP
          - name: metrics
            containerPort: 9091
            protocol: TCP
        livenessProbe:
          tcpSocket:
            port: metrics
          initialDelaySeconds: 90
          periodSeconds: 30
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 5
        readinessProbe:
          httpGet:
            path: /healthz
            port: metrics
          initialDelaySeconds: 90
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 5
        resources:
          limits:
            cpu: 2000m
            memory: 4Gi
          requests:
            cpu: 1000m
            memory: 2Gi
        env:
        volumeMounts:
        - mountPath: /milvus/tools
          name: tools
        - name: milvus-config
          mountPath: /milvus/configs/default.yaml
          subPath: default.yaml
          readOnly: true
        - name: milvus-config
          mountPath: /milvus/configs/user.yaml
          subPath: user.yaml
          readOnly: true
        - name: milvus-data-disk
          mountPath: "/var/lib/milvus"
          subPath:
        - mountPath: /var/lib/milvus/data
          name: disk

      volumes:
      - emptyDir: {}
        name: tools
      - name: milvus-config
        configMap:
          name: ip-milvus
      - name: milvus-data-disk
        persistentVolumeClaim:
          claimName: ip-milvus
      - name: disk
        emptyDir: {}

---

# Source: internal-perplexity/charts/ollama/templates/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
name: ip-ollama
namespace: test
labels:
helm.sh/chart: ollama-1.18.0
app.kubernetes.io/name: ollama
app.kubernetes.io/instance: ip
app.kubernetes.io/version: "0.7.1"
app.kubernetes.io/managed-by: Helm
spec:
replicas: 1
strategy:
type: Recreate
selector:
matchLabels:
app.kubernetes.io/name: ollama
app.kubernetes.io/instance: ip
template:
metadata:
labels:
helm.sh/chart: ollama-1.18.0
app.kubernetes.io/name: ollama
app.kubernetes.io/instance: ip
app.kubernetes.io/version: "0.7.1"
app.kubernetes.io/managed-by: Helm
spec:
serviceAccountName: ip-ollama
securityContext:
{}
containers: - name: ollama
securityContext:
{}
image: "ollama/ollama:latest"
imagePullPolicy: Always
ports: - name: http
containerPort: 11434
protocol: TCP
env: - name: OLLAMA_HOST
value: "0.0.0.0:11434"
envFrom:
args:
resources:
limits:
cpu: 4000m
memory: 8Gi
requests:
cpu: 2000m
memory: 4Gi
volumeMounts: - name: ollama-data
mountPath: /root/.ollama
livenessProbe:
httpGet:
path: /
port: http
initialDelaySeconds: 60
periodSeconds: 10
timeoutSeconds: 5
successThreshold: 1
failureThreshold: 6
readinessProbe:
httpGet:
path: /
port: http
initialDelaySeconds: 30
periodSeconds: 5
timeoutSeconds: 3
successThreshold: 1
failureThreshold: 6
volumes: - name: ollama-data
emptyDir: { }

---

# Source: internal-perplexity/charts/open-webui/charts/pipelines/templates/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
name: open-webui-pipelines
namespace: test
labels:
helm.sh/chart: pipelines-0.7.0
app.kubernetes.io/version: "alpha"
app.kubernetes.io/managed-by: Helm
app.kubernetes.io/instance: ip
app.kubernetes.io/component: open-webui-pipelines
spec:
replicas: 1
selector:
matchLabels:
app.kubernetes.io/instance: ip
app.kubernetes.io/component: open-webui-pipelines
template:
metadata:
labels:
helm.sh/chart: pipelines-0.7.0
app.kubernetes.io/version: "alpha"
app.kubernetes.io/managed-by: Helm
app.kubernetes.io/instance: ip
app.kubernetes.io/component: open-webui-pipelines
spec:
enableServiceLinks: false
automountServiceAccountToken: false
serviceAccountName: open-webui-pipelines
containers: - name: pipelines
image: ghcr.io/open-webui/pipelines:main
imagePullPolicy: Always
ports: - name: http
containerPort: 9099
volumeMounts: - name: data
mountPath: /app/pipelines
env:
tty: true
volumes: - name: data
persistentVolumeClaim:
claimName: open-webui-pipelines

---

# Source: internal-perplexity/charts/milvus/charts/etcd/templates/statefulset.yaml

apiVersion: apps/v1
kind: StatefulSet
metadata:
name: ip-etcd
namespace: test
labels:
app.kubernetes.io/name: etcd
helm.sh/chart: etcd-6.3.3
app.kubernetes.io/instance: ip
app.kubernetes.io/managed-by: Helm
spec:
replicas: 1
selector:
matchLabels:
app.kubernetes.io/name: etcd
app.kubernetes.io/instance: ip
serviceName: ip-etcd-headless
podManagementPolicy: Parallel
updateStrategy:
type: RollingUpdate
template:
metadata:
labels:
app.kubernetes.io/name: etcd
helm.sh/chart: etcd-6.3.3
app.kubernetes.io/instance: ip
app.kubernetes.io/managed-by: Helm
annotations:
spec:

      affinity:
        podAffinity:

        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app.kubernetes.io/name: etcd
                    app.kubernetes.io/instance: ip
                namespaces:
                  - "test"
                topologyKey: kubernetes.io/hostname
              weight: 1
        nodeAffinity:

      securityContext:
        fsGroup: 1001
      serviceAccountName: "default"
      containers:
        - name: etcd
          image: docker.io/milvusdb/etcd:3.5.18-r1
          imagePullPolicy: "IfNotPresent"
          securityContext:
            runAsNonRoot: true
            runAsUser: 1001
          env:
            - name: BITNAMI_DEBUG
              value: "false"
            - name: MY_POD_IP
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
            - name: MY_POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: ETCDCTL_API
              value: "3"
            - name: ETCD_ON_K8S
              value: "yes"
            - name: ETCD_START_FROM_SNAPSHOT
              value: "no"
            - name: ETCD_DISASTER_RECOVERY
              value: "no"
            - name: ETCD_NAME
              value: "$(MY_POD_NAME)"
            - name: ETCD_DATA_DIR
              value: "/bitnami/etcd/data"
            - name: ETCD_LOG_LEVEL
              value: "info"
            - name: ALLOW_NONE_AUTHENTICATION
              value: "yes"
            - name: ETCD_ADVERTISE_CLIENT_URLS
              value: "http://$(MY_POD_NAME).ip-etcd-headless.test.svc.cluster.local:2379"
            - name: ETCD_LISTEN_CLIENT_URLS
              value: "http://0.0.0.0:2379"
            - name: ETCD_INITIAL_ADVERTISE_PEER_URLS
              value: "http://$(MY_POD_NAME).ip-etcd-headless.test.svc.cluster.local:2380"
            - name: ETCD_LISTEN_PEER_URLS
              value: "http://0.0.0.0:2380"
            - name: ETCD_AUTO_COMPACTION_MODE
              value: "revision"
            - name: ETCD_AUTO_COMPACTION_RETENTION
              value: "1000"
            - name: ETCD_QUOTA_BACKEND_BYTES
              value: "4294967296"
            - name: ETCD_HEARTBEAT_INTERVAL
              value: "500"
            - name: ETCD_ELECTION_TIMEOUT
              value: "2500"
          envFrom:
          ports:
            - name: client
              containerPort: 2379
              protocol: TCP
            - name: peer
              containerPort: 2380
              protocol: TCP
          livenessProbe:
            exec:
              command:
                - /opt/bitnami/scripts/etcd/healthcheck.sh
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 10
            successThreshold: 1
            failureThreshold: 5
          readinessProbe:
            exec:
              command:
                - /opt/bitnami/scripts/etcd/healthcheck.sh
            initialDelaySeconds: 60
            periodSeconds: 20
            timeoutSeconds: 10
            successThreshold: 1
            failureThreshold: 5
          resources:
            limits: {}
            requests: {}
          volumeMounts:
            - name: data
              mountPath: /bitnami/etcd
      volumes:

volumeClaimTemplates: - metadata:
name: data
spec:
accessModes: - "ReadWriteOnce"
resources:
requests:
storage: "10Gi"

---

# Source: internal-perplexity/charts/open-webui/templates/workload-manager.yaml

apiVersion: apps/v1
kind: StatefulSet
metadata:
name: open-webui
namespace: test
labels:
helm.sh/chart: open-webui-6.17.0
app.kubernetes.io/version: "0.6.11"
app.kubernetes.io/managed-by: Helm
app.kubernetes.io/instance: ip
app.kubernetes.io/component: open-webui
spec:
replicas: 1
serviceName: open-webui
selector:
matchLabels:
app.kubernetes.io/instance: ip
app.kubernetes.io/component: open-webui
template:
metadata:
labels:
helm.sh/chart: open-webui-6.17.0
app.kubernetes.io/version: "0.6.11"
app.kubernetes.io/managed-by: Helm
app.kubernetes.io/instance: ip
app.kubernetes.io/component: open-webui
spec:
initContainers: - name: copy-app-data
image: ghcr.io/open-webui/open-webui:main
imagePullPolicy: Always
command: ['sh', '-c', 'cp -R -n /app/backend/data/* /tmp/app-data/']
volumeMounts: - name: data
mountPath: /tmp/app-data
enableServiceLinks: false
automountServiceAccountToken: false
serviceAccountName: open-webui
containers: - name: open-webui
image: ghcr.io/open-webui/open-webui:main
imagePullPolicy: Always
ports: - name: http
containerPort: 8080
resources:
limits:
cpu: 1000m
memory: 2Gi
requests:
cpu: 500m
memory: 1Gi
volumeMounts: - name: data
mountPath: /app/backend/data
env: - name: "ENABLE_OLLAMA_API"
value: "False" # If Pipelines is enabled and OpenAI API value is set, use OPENAI_API_BASE_URLS with combined values - name: "OPENAI_API_BASE_URLS"
value: "http://open-webui-pipelines.test.svc.cluster.local:9099;https://api.openai.com/v1" - name: OPENAI_API_KEY
value: 0p3n-w3bu!
tty: true
volumes: - name: data
persistentVolumeClaim:
claimName: open-webui

---

# Source: internal-perplexity/charts/open-webui/templates/ingress.yaml

apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
name: open-webui
namespace: test
labels:
helm.sh/chart: open-webui-6.17.0
app.kubernetes.io/version: "0.6.11"
app.kubernetes.io/managed-by: Helm
app.kubernetes.io/instance: ip
app.kubernetes.io/component: open-webui
annotations:
nginx.ingress.kubernetes.io/ssl-redirect: "false"
spec:
ingressClassName: nginx
rules:

- host: oi.api.meatheadmathematician.com
  http:
  paths:
  - path: /
    pathType: Prefix
    backend:
    service:
    name: open-webui
    port:
    name: http

---

# Source: internal-perplexity/charts/ollama/templates/tests/test-connection.yaml

apiVersion: v1
kind: Pod
metadata:
name: "ip-ollama-test-connection"
namespace: test
labels:
helm.sh/chart: ollama-1.18.0
app.kubernetes.io/name: ollama
app.kubernetes.io/instance: ip
app.kubernetes.io/version: "0.7.1"
app.kubernetes.io/managed-by: Helm
annotations:
"helm.sh/hook": test
spec:
containers: - name: wget
image: busybox
command: ['wget']
args: ['ip-ollama:11434']
restartPolicy: Never
