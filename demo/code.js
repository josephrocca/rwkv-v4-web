
async function pyodideInit() {
  window.pyodide = await loadPyodide();
  await pyodide.loadPackage("micropip");

  // Add tokenizer.json file to Pyodide filesystem:
  let tokenizerJsonText = await fetch("./tokenizer.json").then(r => r.text())
  pyodide.FS.writeFile("/tokenizer.json", tokenizerJsonText, { encoding: "utf8" });

  // Install tokenizer:
  console.log(await pyodide.runPythonAsync(`
  import sys
  print(sys.version)
  import os
  os.environ["TOKENIZERS_PARALLELISM"] = "0" # This is needed because threading doesn't work in Pyodide yet: https://github.com/pyodide/pyodide/issues/2816#issue-1290719241
  import micropip
  await micropip.install('./tokenizers_python-0.11.0-cp310-cp310-emscripten_3_1_14_wasm32.whl')
  from tokenizers import Tokenizer
  tokenizer = Tokenizer.from_file("/tokenizer.json")
  `));
}

function textToTokens(text) {
  pyodide.globals.set("input_text", text);
  pyodide.runPython(`
  encoded = tokenizer.encode(input_text)
  ids = encoded.ids
  tokens = encoded.tokens
  `);
  // pyodide.globals.get('tokens').toJs()
  return pyodide.globals.get('ids').toJs();
}

function tokensToText(tokens) {
  pyodide.globals.set("input_tokens", tokens.join(","));
  pyodide.runPython(`
  input_tokens = [int(x) for x in input_tokens.split(',')]
  decoded = tokenizer.decode(input_tokens)
  `);
  return pyodide.globals.get('decoded');
}

function padLeftWithZeros(arr, size) {
  arr = arr.slice(0);
  
  while (arr.length < size) {
    arr.unshift(0);
  }
  
  return arr;
}

async function createOrtSession(onnxModelBlob, isOrtFile, backend, n_layer, n_embd) {
  ort.env.wasm.proxy = true; // <-- When using wasm, proxy inference via a web worker so it doesn't freeze the main/rendering thread.
  
  if(self.crossOriginIsolated) { // needs to be cross-origin-isolated to use wasm threads. you need to serve this html file with these two headers: https://web.dev/coop-coep/
    ort.env.wasm.numThreads = navigator.hardwareConcurrency / 2;
  }
  
  ort.logLevel = "verbose";
  ort.logLevelInternal = "verbose";

  let sessionOptions = {
    executionProviders: [ backend ],
    graphOptimizationLevel: 'all',
  };
  
  if (isOrtFile) {
    // See here for details: github.com/microsoft/onnxruntime/issues/13445#issuecomment-1430153341
    sessionOptions = {
      executionProviders: [ backend ],
      enableMemPattern: false,
      enableCpuMemArena: false,
      extra: {
        session: {
          disable_prepacking: "1",
          use_device_allocator_for_initializers: "0",
          use_ort_model_bytes_directly: "1",
          use_ort_model_bytes_for_initializers: "1",
        },
      },
    };
  }

  let onnxModelBlobUrl = URL.createObjectURL(onnxModelBlob);
  const session = await ort.InferenceSession.create(onnxModelBlobUrl, sessionOptions);
  URL.revokeObjectURL(onnxModelBlobUrl);

  async function predictText(promptText, numTokensToGenerate=32, streamingCallback=null, abortSignal=undefined, samplingMethod='multinomial', temperature=1.0, topP=0.8, repetitivePenality = 0, show_other_tokens = false) {
  
    let startTime = Date.now();
    
    const xx_att_d = new Float32Array(n_layer*n_embd);
    const aa_att_d = new Float32Array(n_layer*n_embd);
    const bb_att_d = new Float32Array(n_layer*n_embd);
    const pp_att_d = new Float32Array(n_layer*n_embd);
    const xx_ffn_d = new Float32Array(n_layer*n_embd);
    
    pp_att_d.fill(-1e30);
    
    const xx_att = new ort.Tensor('float32', xx_att_d, [n_layer, n_embd]);
    const aa_att = new ort.Tensor('float32', aa_att_d, [n_layer, n_embd]);
    const bb_att = new ort.Tensor('float32', bb_att_d, [n_layer, n_embd]);
    const pp_att = new ort.Tensor('float32', pp_att_d, [n_layer, n_embd]);
    const xx_ffn = new ort.Tensor('float32', xx_ffn_d, [n_layer, n_embd]);
    
    // prepare feeds. use model input names as keys.
    let feeds = { idx: null, xx_att: xx_att, aa_att: aa_att, bb_att: bb_att, pp_att: pp_att, xx_ffn: xx_ffn };
    
    let promptTokens = textToTokens(promptText);
    const origPromptTokensLength = promptTokens.length;
    const ctx = [ promptTokens.shift() ];
    if (streamingCallback) streamingCallback({ token: ctx[0], status: 'Reading prompt', i: 1, outOf: origPromptTokensLength});
    
    const multinomialSampling = samplingMethod === 'multinomial';
    
    if (multinomialSampling) {
      console.log('doing multinomial sampling with temp', temperature, 'and topP', topP, 'and repetitivePenality', repetitivePenality);
    } else {
      console.log('doing greedy sampling with repetitivePenality', repetitivePenality);
    }
    
    
    // feed inputs and run
    for (var i = 0; i < numTokensToGenerate; i++) {
      if(abortSignal && abortSignal.cancelled) {
        break;
      }
      let idx_d = Int32Array.from( padLeftWithZeros(ctx, 1024) );
      let idx = new ort.Tensor('int32', idx_d, [1024]);
      
      feeds.idx = idx;
      
      let results = await session.run(feeds);
      if(abortSignal && abortSignal.cancelled) {
        break;
      }

      let token;
      if (promptTokens.length == 0) {
        let other_tokens;
        if (multinomialSampling) {
          const data = Object.values(results.x.data);
          if (streamingCallback && show_other_tokens) {
            const probs = getMultinomialProbs(applyRepetitionPenalty(data, ctx, repetitivePenality), temperature, topP);
            token = choiceIndex(probs);
            other_tokens = [];
            for (let i = 0; i < probs.length; i++) {
              if(i != token && probs[i] > 0) other_tokens.push(i);
            }
          } else {
            token = npsample(data, temperature, topP, ctx, repetitivePenality);
          }
        } else {
          token = greedySampling(results.x.data, ctx, repetitivePenality);
        }
        
        if (streamingCallback) streamingCallback({ token: token, other_tokens: other_tokens, status: 'Output', i: i+1 - origPromptTokensLength, outOf: numTokensToGenerate - origPromptTokensLength});

      } else {
        token = promptTokens.shift();
        if (streamingCallback) streamingCallback({ token: token, status: 'Reading prompt', i: i+1, outOf: origPromptTokensLength});
      }
      ctx.push(token);
      
      feeds.xx_att = results.xx_att_r;
      feeds.aa_att = results.aa_att_r;
      feeds.bb_att = results.bb_att_r;
      feeds.pp_att = results.pp_att_r;
      feeds.xx_ffn = results.xx_ffn_r;
    }
    
    let timeTaken = Date.now() - startTime;
    console.log(`Finished. Took ${timeTaken}ms`);
    if (streamingCallback) streamingCallback({ status: 'Finished', tokensPerSec: (i+1)/(timeTaken/1000)});
    
    return tokensToText(ctx);
  }
  return {predictText};
}
