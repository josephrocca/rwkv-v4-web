# RWKV-v4 running in the browser
BlinkDL's RWKV-v4 running in the browser. [Thanks to @AXKuhta](https://github.com/BlinkDL/RWKV-LM/issues/7#issuecomment-1221261944) for their work in getting this working!

* **Browser demo**: https://josephrocca.github.io/rwkv-v4-web/demo
* **ONNX conversion notebook**: https://colab.research.google.com/github/josephrocca/rwkv-v4-web/blob/main/RWKV_v4_ONNX_conversion.ipynb
* **Hugging Face repo**: https://huggingface.co/rocca/rwkv-4-pile-web

See the [@BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) repo for more info on RWKV-LM.

**TODO**:
* Optimise loop per AXKuhta's recommendations: https://github.com/BlinkDL/RWKV-LM/issues/7#issuecomment-1227341575
* Speed up the initial-state-computation when the code becomes available: https://github.com/BlinkDL/RWKV-LM/issues/7#issuecomment-1228739721
