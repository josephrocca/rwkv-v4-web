# RWKV-v4 running in the browser
BlinkDL's RWKV-v4 running in the browser. [Thanks to @AXKuhta](https://github.com/BlinkDL/RWKV-LM/issues/7#issuecomment-1221261944) for their work in getting this working!

* **Browser demo**: https://josephrocca.github.io/rwkv-v4-web/demo
* **ONNX conversion notebook**: https://colab.research.google.com/github/josephrocca/rwkv-v4-web/blob/main/RWKV_v4_ONNX_conversion.ipynb
* **Hugging Face repo**: https://huggingface.co/rocca/rwkv-4-pile-web

See the [@BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) repo for more info on RWKV-LM.

# Local Development

## Serving

Note that we require the html file is served with:

```
Cross-Origin-Embedder-Policy: require-corp // or: credentialless
Cross-Origin-Opener-Policy: same-origin
```

Unfortunately this means that we can't use a lot of the simple ways of serving a local html file.  None of vscode live-server extensions work because at
time of writing they don't support custom headers.

One way is:
```sh
# First install npm.  On windows you can use https://github.com/coreybutler/nvm-windows/releases then restart vscode
nvm install 19 && nvm use 19 # If you don't have npm yet
npx servez --shared-array-buffers --local demo 
```

Then open `http://localhost:8080/`

## Loading the model

If you hit the huggingface endpoint too often, it will start failing your downloads.

I recommend downloading the onnx file (e.g. `https://huggingface.co/rocca/rwkv-4-pile-web/resolve/main/169m/rwkv-4-pile-169m-uint8.onnx`) into the `demo/` folder, then using the `load local copy` button.

## Running tests

Run tests with:

```
npx jest
```

**TODO**:
* Optimise loop per AXKuhta's recommendations: https://github.com/BlinkDL/RWKV-LM/issues/7#issuecomment-1227341575
* Speed up the initial-state-computation when the code becomes available: https://github.com/BlinkDL/RWKV-LM/issues/7#issuecomment-1228739721
