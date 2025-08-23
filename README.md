# Auto Batching Proxy

It will automatically batch inference requests from multiple independent users together in a single batch request for efficiency, so that for users the interface looks like individual requests, but internally it is handled as a batch 
request, essentially it provide a REST API wrapper around some inference service like https://github.com/huggingface/text-embeddings-inference


Proxy server is configured with following parameters:

_Max Wait Time_ - maximal time user request can wait for other requests to be accumulated in a batch  
_Max Batch Size_ - maximal number of requests that can be accumulated in a batch.

## Setup Inference Service
First, try running inference service in a container with `--model-id nomic-ai/nomic-embed-text-v1.5`
```
docker run --rm -it -p 8080:80 --pull always \
    ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
--model-id nomic-ai/nomic-embed-text-v1.5
```
if it fails to start, then try with some other alternatives. Currently, code is functional for  
`--model-id sentence-transformers/all-MiniLM-L6-v2` & `--model-id sentence-transformers/all-mpnet-base-v2`
Check [nomic-embed-text-v1.5_FAILED.png](screenshots/run_model_status/nomic-embed-text-v1.5_FAILED.png) & others 
inside [/screenshots](./screenshots)

```
docker run --rm -it -p 8080:80 --pull always \
  ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
  --model-id sentence-transformers/all-MiniLM-L6-v2
```
Note: [Backend does not support a batch size > 8](screenshots/run_model_status/max_batch_size.png)
but our proxy will respect this config param & will not send requests (as well as max inputs, which is 32 for `all-MiniLM-L6-v2`) more than supported batch size. 

## Setup Proxy service
- either run `cargo run` at root of the project, it will launch Rocket [with default configuration params](./screenshots/cargo/cargo_run.png)
- or otherwise [with custom params](./screenshots/cargo/cargo_run_with_params.png) like `RUST_LOG=INFO cargo run -- --max-batch-size 100 --max-wait-time-ms 3000`


**[Unit tests](https://doc.rust-lang.org/book/ch11-03-test-organization.html#unit-tests)**   
Relevant unit tests are provided inside source code files inside `/src`

**[Integration tests](https://doc.rust-lang.org/book/ch11-03-test-organization.html#integration-tests)**  
Check the `/tests` folder, code is covered with various scenarios.

Run all tests via `cargo test`. Currently, tests are verified to be passed against  
`--model-id sentence-transformers/all-MiniLM-L6-v2` & ` --model-id sentence-transformers/all-mpnet-base-v2`
& they also explain how/why which part of code was written for which particular use case.

Use the following simple CURL commands for quick testing
**via CURL**
- for inference
```
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Hello world"]}'
```
- for proxy
```
curl -X POST http://localhost:3000/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Hello", "World"]}'
```
to verify proxy is working for multiple concurrent requests 
```
cd scripts 
./proxy_concurrent_calls.sh
```

**Benchmark test results:**
Following output is taken from 
```
cargo test test_compare_single_input_inference_service_vs_auto_batching_proxy_with_30_separate_requests -- --nocapture
```
[--nocapture](https://doc.rust-lang.org/cargo/commands/cargo-test.html#display-options) will recover display output 

```
Timing Summary:
Requests        Direct             Proxy     
  1          13.511366ms        65.28781ms
  5          52.58287ms         73.557987ms
 10         101.085006ms        75.359103ms
 25         382.281018ms        117.845833ms
 30         364.412884ms        71.834355ms
 50         607.902053ms        126.407797ms
 75              N/A            160.6118ms
 100             N/A            205.71627ms
```
